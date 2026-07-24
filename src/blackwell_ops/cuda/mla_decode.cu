// MLA decode kernel with global-memory streaming for SM120
// Computes: O = softmax(Q' @ c_KV^T / sqrt(L)) @ c_KV
// Q': [B, H, L] (pre-absorbed query), c_KV: [B, N, L] (latent cache)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_THREADS 256
#define TILE_N 64

//--------------------------------------------------------------------------------
// Decode attention over latent cache (single query per head)
// Streams KV tiles from global memory to avoid 48KB shared-memory limit.
//--------------------------------------------------------------------------------
template <typename T>
__global__ void mla_decode_kernel(const T* __restrict__ Q, const T* __restrict__ cKV,
                                  T* __restrict__ O, int B, int H, int N, int L,
                                  float scale) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    const T* q = Q + ((long)b * H + h) * L;
    const T* ckv = cKV + (long)b * N * L;
    T* o = O + ((long)b * H + h) * L;

    extern __shared__ float smem[];
    float* sScore = smem;          // [TILE_N]
    float* sQ = smem + TILE_N;     // [L]
    float* sOut = sQ + L;          // [L]
    float* sMax = sOut + L;        // [1]
    float* sSum = sMax + 1;        // [1]

    for (int i = tid; i < L; i += blockDim.x) {
        sQ[i] = static_cast<float>(q[i]);
        sOut[i] = 0.0f;
    }
    if (tid == 0) {
        *sMax = -CUDART_INF_F;
        *sSum = 0.0f;
    }
    __syncthreads();

    for (int n0 = 0; n0 < N; n0 += TILE_N) {
        int tile_n = min(TILE_N, N - n0);

        // compute scores for this tile, reading KV directly from global memory
        for (int n = tid; n < tile_n; n += blockDim.x) {
            float acc = 0.0f;
            const T* kv = ckv + (n0 + n) * L;
            for (int l = 0; l < L; ++l) {
                acc += sQ[l] * static_cast<float>(kv[l]);
            }
            sScore[n] = acc * scale;
        }
        __syncthreads();

        // block reduce max
        float m_tile = -CUDART_INF_F;
        for (int n = tid; n < tile_n; n += blockDim.x) {
            m_tile = fmaxf(m_tile, sScore[n]);
        }
        __shared__ float red[BLOCK_THREADS];
        red[tid] = m_tile;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        m_tile = red[0];

        float m_prev = *sMax;
        float m_new = fmaxf(m_prev, m_tile);
        float scale_prev = expf(m_prev - m_new);
        *sMax = m_new;
        *sSum = *sSum * scale_prev;

        for (int l = tid; l < L; l += blockDim.x) {
            sOut[l] *= scale_prev;
        }
        __syncthreads();

        for (int n = 0; n < tile_n; ++n) {
            float p = expf(sScore[n] - m_new);
            if (tid == 0) *sSum += p;
            const T* kv = ckv + (n0 + n) * L;
            for (int l = tid; l < L; l += blockDim.x) {
                sOut[l] += p * static_cast<float>(kv[l]);
            }
        }
        __syncthreads();
    }

    for (int l = tid; l < L; l += blockDim.x) {
        o[l] = static_cast<T>(sOut[l] / *sSum);
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch (registered in basic_ops.cu's PYBIND11_MODULE)
//--------------------------------------------------------------------------------
torch::Tensor mla_decode(torch::Tensor Q, torch::Tensor cKV) {
    TORCH_CHECK(Q.is_cuda() && cKV.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(Q.dim() == 3 && cKV.dim() == 3, "Q [B,H,L], cKV [B,N,L]");
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int L = Q.size(2);
    const int N = cKV.size(1);
    TORCH_CHECK(cKV.size(0) == B && cKV.size(2) == L, "shape mismatch");

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)L);
    dim3 grid(B, H);

    size_t smem = (TILE_N + L + L + 2) * sizeof(float);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(Q.scalar_type(), "mla_decode", [&] {
        mla_decode_kernel<scalar_t><<<grid, BLOCK_THREADS, smem>>>(
            Q.data_ptr<scalar_t>(), cKV.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(),
            B, H, N, L, scale);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
