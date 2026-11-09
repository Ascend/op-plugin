// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
#if VERSION_BETWEEN(V2R14, VERSION_NEWEST)
namespace {
// Input checks. The messages match the CPU/CUDA side (aten::native::squareCheckInputs /
// checkFloatingOrComplex) so that generic OpInfo tests can compare across backends.
void check_linalg_matrix_sqrth_inputs(const at::Tensor& self)
{
    constexpr int64_t kMatrixDim = 2;
    TORCH_CHECK(
        self.dim() >= kMatrixDim,
        "linalg.matrix_sqrth: The input tensor A must have at least 2 dimensions.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        self.sym_size(-1) == self.sym_size(-2),
        "linalg.matrix_sqrth: A must be batches of square matrices, but they are ",
        self.sym_size(-2),
        " by ",
        self.sym_size(-1),
        " matrices",
        OPS_ERROR(ErrCode::PARAM));
    // aclnnSvd accepts 2-8 dims. The reconstruction below also calls aclnnMatmul, which accepts
    // 1-6 dims except on Atlas A2/A3 (1-8), so 7-8 dim inputs additionally depend on the product.
    constexpr int64_t kMaxMatrixDim = 8;
    TORCH_CHECK(
        self.dim() <= kMaxMatrixDim,
        "linalg.matrix_sqrth: The input tensor A must have at most 8 dimensions, but it has ",
        self.dim(),
        OPS_ERROR(ErrCode::PARAM));

    const auto dtype = self.scalar_type();
    TORCH_CHECK(
        at::isFloatingType(dtype) || at::isComplexType(dtype),
        "linalg.matrix_sqrth: Expected a floating point or complex tensor as input. Got ",
        dtype,
        OPS_ERROR(ErrCode::TYPE));
    // Same as CPU/CUDA: upstream calls checkFloatingOrComplex(..., allow_low_precision_dtypes=false),
    // so half/bfloat16 are rejected instead of being upcast.
    TORCH_CHECK(
        dtype != at::kHalf && dtype != at::kBFloat16,
        "linalg.matrix_sqrth: Low precision dtypes not supported. Got ",
        dtype,
        OPS_ERROR(ErrCode::TYPE));
    // Only float32 is enabled here: SVD and matmul support different dtype ranges and their
    // intersection is float32 alone (aclnnSvd: FLOAT/DOUBLE/COMPLEX64/COMPLEX128, aclnnMatmul:
    // FLOAT16/BFLOAT16/FLOAT32), so float64/complex fail explicitly instead of falling back to CPU.
    TORCH_CHECK(
        dtype == at::kFloat,
        "linalg.matrix_sqrth: float64/complex inputs are not supported on the NPU backend yet, got ",
        dtype,
        OPS_ERROR(ErrCode::TYPE));
}

// Recover the signed eigenvalues from the SVD factors of a symmetric matrix:
// for symmetric A, lambda_i = sigma_i * (u_i dot v_i).
// This step is required: SVD returns non-negative sigma and a negative eigenvalue direction gives
// u_i = -v_i, so taking sqrt of sigma directly would produce -sqrt(|lambda|) there, which differs
// from the 0 produced by upstream's clamp_min(0).
at::Tensor signed_eigvals(const at::Tensor& s, const at::Tensor& u, const at::Tensor& vh)
{
    auto signs = (u * vh.transpose(-2, -1)).sum(-2);
    return s * signs;
}

// Core computation of the principal square root, shared by the functional and out variants.
at::Tensor linalg_matrix_sqrth_impl(const at::Tensor& self)
{
    // Empty tensor (n == 0 or an empty batch): the output has the same shape, return a copy.
    if (self.numel() == 0) {
        return self.clone();
    }

    // Read only the lower triangle and rebuild the symmetric matrix (upstream semantics: the input
    // is assumed to be Hermitian and only its lower triangle is used).
    auto a = self.tril() + self.tril(-1).transpose(-2, -1);

    // Use SVD instead of the CPU/CUDA eigh path (CANN has no Hermitian eigendecomposition op):
    // A^(1/2) = U diag(sqrt(clamp(lambda, 0))) V^T, where zero/negative eigenvalues contribute 0.
    // For positive-definite/positive-semidefinite input this matches upstream
    // eigvals.clamp_min(0).sqrt(). Known limitation: if the spectrum contains both lambda and
    // -lambda (indefinite input), the singular values are repeated and the SVD basis is not unique,
    // so the result depends on the implementation.
    // Call the op_api:: version directly to reuse the adapted aclnnSvd and to avoid building a
    // second autograd graph inside the kernel.
    auto svd = op_api::_linalg_svd(a, /*full_matrices=*/false, /*compute_uv=*/true, c10::nullopt);
    const auto& u = std::get<0>(svd);
    const auto& s = std::get<1>(svd);
    const auto& vh = std::get<2>(svd);

    auto sqrt_eigvals = signed_eigvals(s, u, vh).clamp_min(0).sqrt();
    auto result = at::matmul(u * sqrt_eigvals.unsqueeze(-2), vh);
    // Same as upstream: the reconstruction is symmetric up to roundoff, so symmetrize it
    // explicitly to guarantee X == X^T.
    return 0.5 * (result + result.transpose(-2, -1));
}
} // namespace

at::Tensor linalg_matrix_sqrth(const at::Tensor& self)
{
    check_linalg_matrix_sqrth_inputs(self);
    return linalg_matrix_sqrth_impl(self);
}

at::Tensor& linalg_matrix_sqrth_out(const at::Tensor& self, at::Tensor& out)
{
    check_linalg_matrix_sqrth_inputs(self);
    at_npu::native::OpPreparation::check_tensor({self}, out, self.scalar_type(), self.sizes());
    auto result = linalg_matrix_sqrth_impl(self);
    out.copy_(result);
    return out;
}
#endif
} // namespace op_api
