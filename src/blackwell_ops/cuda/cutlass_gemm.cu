// Minimal CUTLASS FP16 GEMM for SM120 comparison (via Sm89 PTX JIT)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, LayoutA,
    cutlass::half_t, LayoutB,
    cutlass::half_t, LayoutC,
    float>;

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf, "fp16 required");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "shape mismatch");

    auto C = torch::empty({M, N}, A.options());

    CutlassGemm gemm_op;
    cutlass::Status status = gemm_op({
        {M, N, K},
        {(cutlass::half_t*)A.data_ptr<at::Half>(), K},
        {(cutlass::half_t*)B.data_ptr<at::Half>(), K},
        {(cutlass::half_t*)C.data_ptr<at::Half>(), N},
        {(cutlass::half_t*)C.data_ptr<at::Half>(), N},
        {1.0f, 0.0f},
    });
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
