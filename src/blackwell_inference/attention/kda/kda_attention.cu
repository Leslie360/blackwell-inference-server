// Kernel Deep Attention (KDA) — raw CUDA fused causal softmax attention.
// FlashAttention-style online softmax with shared-memory tiles.
// Supports FP16/BF16 inputs, FP32 accumulation, head_dim=64.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int D  = 64;

//--------------------------------------------------------------------------------
// Forward kernel
//--------------------------------------------------------------------------------
template <typename T>
__global__ void kda_fwd_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int B, int H, int N,
    int stride_qb, int stride_qh, int stride_qn,
    int stride_kb, int stride_kh, int stride_kn,
    int stride_vb, int stride_vh, int stride_vn,
    int stride_ob, int stride_oh, int stride_on,
    float scale)
{
    const int batch = blockIdx.x / H;
    const int head  = blockIdx.x % H;
    const int q_tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int row = tid;          // each thread owns one query row in the tile

    const int q_row0 = q_tile * Br;
    const int q_row_global = q_row0 + row;
    const int n_tiles = (N + Bc - 1) / Bc;

    // shared memory layout: Q[Br][D], K[Bc][D], V[Bc][D], S[Br][Bc+1] (padded)
    extern __shared__ char smem[];
    T* sQ = reinterpret_cast<T*>(smem);
    T* sK = sQ + Br * D;
    T* sV = sK + Bc * D;
    float* sS = reinterpret_cast<float*>(sV + Bc * D);

    const T* q_base = Q + batch * stride_qb + head * stride_qh;
    const T* k_base = K + batch * stride_kb + head * stride_kh;
    const T* v_base = V + batch * stride_vb + head * stride_vh;
    T* o_base = O + batch * stride_ob + head * stride_oh;

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;
    float o_i[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) o_i[d] = 0.0f;

    // load Q tile (cooperative)
    for (int idx = tid; idx < Br * D; idx += blockDim.x) {
        int r = idx / D;
        int d = idx % D;
        int qr = q_row0 + r;
        sQ[r * D + d] = (qr < N) ? q_base[qr * stride_qn + d] : static_cast<T>(0.0f);
    }
    __syncthreads();

    // iterate KV tiles
    for (int j = 0; j < n_tiles; ++j) {
        const int kv_row0 = j * Bc;

        // load K and V tiles
        for (int idx = tid; idx < Bc * D; idx += blockDim.x) {
            int r = idx / D;
            int d = idx % D;
            int kr = kv_row0 + r;
            sK[r * D + d] = (kr < N) ? k_base[kr * stride_kn + d] : static_cast<T>(0.0f);
            sV[r * D + d] = (kr < N) ? v_base[kr * stride_vn + d] : static_cast<T>(0.0f);
        }
        __syncthreads();

        // compute S tile [Br][Bc] into shared memory, one element per thread iteration
        for (int idx = tid; idx < Br * Bc; idx += blockDim.x) {
            int r = idx / Bc;
            int c = idx % Bc;
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                acc += static_cast<float>(sQ[r * D + d]) * static_cast<float>(sK[c * D + d]);
            }
            acc *= scale;
            int kv_global = kv_row0 + c;
            int qr_global = q_row0 + r;
            if (kv_global > qr_global) acc = -CUDART_INF_F;
            sS[r * (Bc + 1) + c] = acc;
        }
        __syncthreads();

        // each thread processes its assigned row r = row
        if (q_row_global < N) {
            // row max
            float m_prev = m_i;
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                m_i = fmaxf(m_i, sS[row * (Bc + 1) + c]);
            }

            float l_scale = expf(m_prev - m_i);
            l_i *= l_scale;
            #pragma unroll
            for (int d = 0; d < D; ++d) o_i[d] *= l_scale;

            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                float p = expf(sS[row * (Bc + 1) + c] - m_i);
                l_i += p;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    o_i[d] += p * static_cast<float>(sV[c * D + d]);
                }
            }
        }
        __syncthreads();
    }

    // write output
    if (q_row_global < N) {
        T* o_ptr = o_base + q_row_global * stride_on;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            o_ptr[d] = static_cast<T>(o_i[d] / l_i);
        }
    }
}

//--------------------------------------------------------------------------------
// C++ dispatch
//--------------------------------------------------------------------------------
torch::Tensor kda_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(), "Q,K,V must have same dtype");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D_in = Q.size(3);
    TORCH_CHECK(D_in == D, "KDA kernel currently supports head_dim=64");

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    const dim3 blocks(B * H, (N + Br - 1) / Br);
    const int threads = Br;
    const size_t smem = (Br * D + Bc * D + Bc * D) * sizeof(at::Half) + Br * (Bc + 1) * sizeof(float);

    AT_DISPATCH_REDUCED_FLOATING_TYPES(Q.scalar_type(), "kda_forward", [&] {
        kda_fwd_kernel<scalar_t><<<blocks, threads, smem>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(),
            B, H, N,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            scale);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kda_forward, "KDA forward");
}
