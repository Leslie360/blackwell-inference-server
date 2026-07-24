// LoRA delta computation with shared-memory tiling for SM120
// delta = B @ A * scaling
// A: [r, K], B: [N, r] -> delta: [N, K]

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREADS 256

//--------------------------------------------------------------------------------
// Tiled LoRA delta kernel
// Each block computes a [TILE_M, TILE_N] tile of delta.
//--------------------------------------------------------------------------------
__global__ void lora_delta_tiled_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                        half* __restrict__ delta, int N, int K, int r,
                                        float scaling) {
    __shared__ float As[TILE_K][TILE_N];
    __shared__ float Bs[TILE_M][TILE_K];

    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;
    int tid = threadIdx.x;

    float acc[TILE_M / 16][TILE_N / 16];
    #pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i)
        #pragma unroll
        for (int j = 0; j < TILE_N / 16; ++j)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < r; k0 += TILE_K) {
        // load A tile [TILE_K, TILE_N]
        for (int idx = tid; idx < TILE_K * TILE_N; idx += THREADS) {
            int i = idx / TILE_N;
            int j = idx % TILE_N;
            int kg = k0 + i;
            int ng = block_n + j;
            As[i][j] = (kg < r && ng < K) ? __half2float(A[kg * K + ng]) : 0.0f;
        }
        // load B tile [TILE_M, TILE_K]
        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS) {
            int i = idx / TILE_K;
            int j = idx % TILE_K;
            int mg = block_m + i;
            int kg = k0 + j;
            Bs[i][j] = (mg < N && kg < r) ? __half2float(B[mg * r + kg]) : 0.0f;
        }
        __syncthreads();

        // compute
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < TILE_M / 16; ++i) {
                #pragma unroll
                for (int j = 0; j < TILE_N / 16; ++j) {
                    int m = tid / 16 + i * 16;
                    int n = tid % 16 + j * 16;
                    acc[i][j] += Bs[m][k] * As[k][n];
                }
            }
        }
        __syncthreads();
    }

    // write output
    #pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < TILE_N / 16; ++j) {
            int m = block_m + tid / 16 + i * 16;
            int n = block_n + tid % 16 + j * 16;
            if (m < N && n < K) {
                delta[m * K + n] = __float2half(acc[i][j] * scaling);
            }
        }
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch (registered in basic_ops.cu's PYBIND11_MODULE)
//--------------------------------------------------------------------------------
torch::Tensor lora_delta_tiled(torch::Tensor A, torch::Tensor B, double scaling) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf, "A and B must be fp16");
    const int r = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == r, "B must be [N, r]");

    auto delta = torch::empty({N, K}, A.options());
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    lora_delta_tiled_kernel<<<grid, THREADS>>>(
        (const half*)A.data_ptr<at::Half>(), (const half*)B.data_ptr<at::Half>(),
        (half*)delta.data_ptr<at::Half>(), N, K, r, (float)scaling);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return delta;
}
