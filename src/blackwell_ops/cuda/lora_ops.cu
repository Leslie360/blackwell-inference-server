// LoRA delta computation and weight fusion kernels for SM120

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_THREADS 256

//--------------------------------------------------------------------------------
// Compute LoRA delta weight: delta = B @ A * scaling
// A: [r, K], B: [N, r] -> delta: [N, K]
// Simple tiled implementation, one thread per output element.
//--------------------------------------------------------------------------------
__global__ void lora_delta_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                  half* __restrict__ delta, int N, int K, int r, float scaling) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || k >= K) return;

    float acc = 0.0f;
    for (int j = 0; j < r; ++j) {
        acc += __half2float(B[n * r + j]) * __half2float(A[j * K + k]);
    }
    delta[n * K + k] = __float2half(acc * scaling);
}

//--------------------------------------------------------------------------------
// Add delta to weight in-place: W += delta
//--------------------------------------------------------------------------------
__global__ void add_delta_kernel(half* __restrict__ W, const half* __restrict__ delta, long numel) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < numel; i += stride) {
        W[i] = __float2half(__half2float(W[i]) + __half2float(delta[i]));
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch (registered in basic_ops.cu's PYBIND11_MODULE)
//--------------------------------------------------------------------------------
torch::Tensor lora_delta(torch::Tensor A, torch::Tensor B, double scaling) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf, "A and B must be fp16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    const int r = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == r, "B must be [N, r]");

    auto delta = torch::empty({N, K}, A.options());
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    lora_delta_kernel<<<grid, block>>>(
        (const half*)A.data_ptr<at::Half>(), (const half*)B.data_ptr<at::Half>(),
        (half*)delta.data_ptr<at::Half>(), N, K, r, (float)scaling);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return delta;
}

void add_delta_(torch::Tensor W, torch::Tensor delta) {
    TORCH_CHECK(W.is_cuda() && delta.is_cuda(), "W and delta must be CUDA");
    TORCH_CHECK(W.scalar_type() == at::kHalf && delta.scalar_type() == at::kHalf, "must be fp16");
    TORCH_CHECK(W.numel() == delta.numel(), "shape mismatch");
    long numel = W.numel();
    int blocks = (numel + BLOCK_THREADS - 1) / BLOCK_THREADS;
    add_delta_kernel<<<blocks, BLOCK_THREADS>>>(
        (half*)W.data_ptr<at::Half>(), (const half*)delta.data_ptr<at::Half>(), numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
