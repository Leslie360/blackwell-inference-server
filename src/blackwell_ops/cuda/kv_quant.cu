// KV cache INT8 quantization kernels for SM120

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_THREADS 256

//--------------------------------------------------------------------------------
// Per-channel symmetric INT8 quantization
// x: [C, N] fp16 (C channels, each of length N)
// q: [C, N] int8
// scale: [C] fp32
//--------------------------------------------------------------------------------
__global__ void quantize_per_channel_kernel(const half* __restrict__ x, int8_t* __restrict__ q,
                                            float* __restrict__ scale, int C, long N) {
    int c = blockIdx.x;
    const half* x_c = x + (long)c * N;
    int8_t* q_c = q + (long)c * N;

    // find max abs in channel
    float max_abs = 0.0f;
    for (long i = threadIdx.x; i < N; i += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(__half2float(x_c[i])));
    }
    __shared__ float smem[BLOCK_THREADS];
    smem[threadIdx.x] = max_abs;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float s = smem[0] / 127.0f;
    if (threadIdx.x == 0) scale[c] = s;

    // quantize
    for (long i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(x_c[i]) / s;
        q_c[i] = (int8_t)max(-127, min(127, (int)lrintf(v)));
    }
}

//--------------------------------------------------------------------------------
// Per-channel INT8 dequantization
//--------------------------------------------------------------------------------
__global__ void dequantize_per_channel_kernel(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                              half* __restrict__ x, int C, long N) {
    int c = blockIdx.x;
    const int8_t* q_c = q + (long)c * N;
    half* x_c = x + (long)c * N;
    float s = scale[c];

    for (long i = threadIdx.x; i < N; i += blockDim.x) {
        x_c[i] = __float2half((float)q_c[i] * s);
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch (registered in basic_ops.cu's PYBIND11_MODULE)
//--------------------------------------------------------------------------------
std::vector<torch::Tensor> quantize_kv_int8(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "x must be CUDA fp16");
    TORCH_CHECK(x.dim() == 2, "x must be [C, N]");
    const int C = x.size(0);
    const long N = x.size(1);

    auto q = torch::empty({C, N}, x.options().dtype(torch::kChar));
    auto scale = torch::empty({C}, x.options().dtype(torch::kFloat));

    quantize_per_channel_kernel<<<C, BLOCK_THREADS>>>(
        (const half*)x.data_ptr<at::Half>(), q.data_ptr<int8_t>(), scale.data_ptr<float>(), C, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, scale};
}

torch::Tensor dequantize_kv_int8(torch::Tensor q, torch::Tensor scale) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kChar, "q must be CUDA int8");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat, "scale must be CUDA fp32");
    TORCH_CHECK(q.dim() == 2, "q must be [C, N]");
    const int C = q.size(0);
    const long N = q.size(1);

    auto x = torch::empty({C, N}, q.options().dtype(torch::kHalf));
    dequantize_per_channel_kernel<<<C, BLOCK_THREADS>>>(
        q.data_ptr<int8_t>(), scale.data_ptr<float>(), (half*)x.data_ptr<at::Half>(), C, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return x;
}
