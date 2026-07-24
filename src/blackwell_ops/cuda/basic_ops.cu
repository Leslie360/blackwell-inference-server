// blackwell-ops CUDA kernels for SM120
// RMSNorm, RoPE, SwiGLU, fused residual+RMSNorm

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_THREADS 256

//--------------------------------------------------------------------------------
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// One block per row.
//--------------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int N, float eps) {
    __shared__ float smem[BLOCK_THREADS];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* x_row = x + (long)row * N;
    T* y_row = y + (long)row * N;

    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = static_cast<float>(x_row[i]);
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rstd = 1.0f / sqrtf(smem[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        y_row[i] = static_cast<T>(static_cast<float>(x_row[i]) * rstd * static_cast<float>(w[i]));
    }
}

//--------------------------------------------------------------------------------
// Fused residual + RMSNorm: y = rmsnorm(x + residual)
//--------------------------------------------------------------------------------
template <typename T>
__global__ void fused_residual_rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ residual,
                                              const T* __restrict__ w, T* __restrict__ y,
                                              int N, float eps) {
    __shared__ float smem[BLOCK_THREADS];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* x_row = x + (long)row * N;
    const T* r_row = residual + (long)row * N;
    T* y_row = y + (long)row * N;

    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = static_cast<float>(x_row[i]) + static_cast<float>(r_row[i]);
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rstd = 1.0f / sqrtf(smem[0] / N + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float v = static_cast<float>(x_row[i]) + static_cast<float>(r_row[i]);
        y_row[i] = static_cast<T>(v * rstd * static_cast<float>(w[i]));
    }
}

//--------------------------------------------------------------------------------
// RoPE: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
// Grid: (B, H, N), each thread handles one pair.
//--------------------------------------------------------------------------------
template <typename T>
__global__ void rope_kernel(const T* __restrict__ x, const float* __restrict__ cos,
                            const float* __restrict__ sin, T* __restrict__ y,
                            int H, int N, int D) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int n = blockIdx.z;
    int tid = threadIdx.x;
    int half = D / 2;

    const T* x_ptr = x + ((long)b * H * N + (long)h * N + n) * D;
    T* y_ptr = y + ((long)b * H * N + (long)h * N + n) * D;

    for (int d = tid; d < half; d += blockDim.x) {
        float x1 = static_cast<float>(x_ptr[d]);
        float x2 = static_cast<float>(x_ptr[d + half]);
        float c = cos[n * half + d];
        float s = sin[n * half + d];
        y_ptr[d] = static_cast<T>(x1 * c - x2 * s);
        y_ptr[d + half] = static_cast<T>(x1 * s + x2 * c);
    }
}

//--------------------------------------------------------------------------------
// SwiGLU: y = silu(gate) * up
// Elementwise.
//--------------------------------------------------------------------------------
template <typename T>
__global__ void swiglu_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                              T* __restrict__ y, long numel) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < numel; i += stride) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        float silu = g / (1.0f + expf(-g));
        y[i] = static_cast<T>(silu * u);
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch
//--------------------------------------------------------------------------------
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    const int M = x.size(0);
    const int N = x.size(1);
    auto y = torch::empty_like(x);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "rmsnorm", [&] {
        rmsnorm_kernel<scalar_t><<<M, BLOCK_THREADS>>>(
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), N, (float)eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor fused_residual_rmsnorm(torch::Tensor x, torch::Tensor residual, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    const int M = x.size(0);
    const int N = x.size(1);
    auto y = torch::empty_like(x);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "fused_residual_rmsnorm", [&] {
        fused_residual_rmsnorm_kernel<scalar_t><<<M, BLOCK_THREADS>>>(
            x.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(), N, (float)eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, N, D]");
    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    auto y = torch::empty_like(x);
    dim3 grid(B, H, N);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "rope", [&] {
        rope_kernel<scalar_t><<<grid, BLOCK_THREADS>>>(
            x.data_ptr<scalar_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
            y.data_ptr<scalar_t>(), H, N, D);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor swiglu(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(gate.numel() == up.numel(), "gate and up must have same numel");
    auto y = torch::empty_like(gate);
    long numel = gate.numel();
    int blocks = (numel + BLOCK_THREADS - 1) / BLOCK_THREADS;
    AT_DISPATCH_REDUCED_FLOATING_TYPES(gate.scalar_type(), "swiglu", [&] {
        swiglu_kernel<scalar_t><<<blocks, BLOCK_THREADS>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), numel);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// declared in kv_quant.cu
std::vector<torch::Tensor> quantize_kv_int8(torch::Tensor x);
torch::Tensor dequantize_kv_int8(torch::Tensor q, torch::Tensor scale);

// declared in lora_ops.cu
torch::Tensor lora_delta(torch::Tensor A, torch::Tensor B, double scaling);
void add_delta_(torch::Tensor W, torch::Tensor delta);

// declared in lora_tiled.cu
torch::Tensor lora_delta_tiled(torch::Tensor A, torch::Tensor B, double scaling);

// declared in mla_decode.cu
torch::Tensor mla_decode(torch::Tensor Q, torch::Tensor cKV);

// declared in cutlass_gemm.cu
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
    m.def("fused_residual_rmsnorm", &fused_residual_rmsnorm, "Fused residual + RMSNorm (CUDA)");
    m.def("rope", &rope, "RoPE (CUDA)");
    m.def("swiglu", &swiglu, "SwiGLU (CUDA)");
    m.def("quantize_kv_int8", &quantize_kv_int8, "Quantize KV cache to INT8 (CUDA)");
    m.def("dequantize_kv_int8", &dequantize_kv_int8, "Dequantize KV cache from INT8 (CUDA)");
    m.def("lora_delta", &lora_delta, "Compute LoRA delta weight (CUDA)");
    m.def("lora_delta_tiled", &lora_delta_tiled, "Compute LoRA delta weight with tiling (CUDA)");
    m.def("add_delta_", &add_delta_, "Add LoRA delta to weight in-place (CUDA)");
    m.def("mla_decode", &mla_decode, "MLA decode with weight absorption (CUDA)");
    m.def("cutlass_gemm", &cutlass_gemm, "CUTLASS FP16 GEMM (CUDA)");
}
