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
// Mirrors upstream linalg_matrix_sqrth_differential: lambda_i = sigma_i * (u_i dot v_i).
at::Tensor signed_eigvals_backward(const at::Tensor& s, const at::Tensor& u, const at::Tensor& vh)
{
    auto signs = (u * vh.transpose(-2, -1)).sum(-2);
    return s * signs;
}
} // namespace

// Backward of torch.linalg.matrix_sqrth (shared by VJP and JVP).
//
// Backward operator: dX = L_A(dA), where X = A^(1/2). This is equivalent to the CPU/CUDA
// linalg_matrix_sqrth_differential (Daleckii-Krein/Loewner formula):
//     L_A(E) = Q (W . (Q^T sym(E) Q)) Q^T,  W_ij = 1 / (sqrt(lambda_i) + sqrt(lambda_j))
// The only difference is that the eigenpairs (Q, lambda) come from SVD instead of eigh:
//     for symmetric A, A v_i = sigma_i u_i and A v_i = lambda_i v_i,
//     so Q = U and lambda = sigma * (u_i dot v_i) give the same decomposition.
// W is real symmetric and Q is orthogonal, so the operator is self-adjoint; one function
// therefore serves both VJP and JVP.
//
// Constraints, same as upstream: the input must be positive definite; at zero/negative
// eigenvalues sqrt(lambda) = 0 and the denominator may vanish, which leaves the backward
// undefined (upstream behaves the same way); second order derivatives are not guaranteed.
at::Tensor npu_linalg_matrix_sqrth_backward(const at::Tensor& grad, const at::Tensor& self)
{
    // Same as upstream linalg_matrix_sqrth_differential: an undefined cotangent takes no part in
    // the computation and yields an empty tensor, which the autograd engine treats as zero
    // (gradcheck's check_undefined_grad passes one in).
    if (!grad.defined()) {
        return {};
    }

    constexpr int64_t kMatrixDim = 2;
    TORCH_CHECK(
        self.dim() >= kMatrixDim && self.size(-1) == self.size(-2),
        "npu_linalg_matrix_sqrth_backward: A must be a batch of square matrices.",
        OPS_ERROR(ErrCode::PARAM));
    constexpr int64_t kMaxMatrixDim = 8;
    TORCH_CHECK(
        self.dim() <= kMaxMatrixDim,
        "npu_linalg_matrix_sqrth_backward: A must have at most 8 dimensions, got ",
        self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        self.scalar_type() == at::kFloat,
        "npu_linalg_matrix_sqrth_backward: only float32 is supported on the NPU backend, got ",
        self.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));

    // Empty tensor (n == 0 or an empty batch): the gradient has the same shape as the input.
    if (self.numel() == 0) {
        return grad.clone();
    }

    // self is the A already handled with the forward semantics (lower triangle only, assumed
    // symmetric); rebuild the same matrix here.
    auto a = self.tril() + self.tril(-1).transpose(-2, -1);

    // Take the (sign-restored) eigenvalues from SVD; for positive-definite/positive-semidefinite
    // input this is equivalent to the CPU/CUDA eigh path. Known limitation: when the spectrum
    // contains both lambda and -lambda the SVD basis is not unique and the result differs from CPU.
    auto svd = op_api::_linalg_svd(a, /*full_matrices=*/false, /*compute_uv=*/true, c10::nullopt);
    const auto& u = std::get<0>(svd);
    const auto& s = std::get<1>(svd);
    const auto& vh = std::get<2>(svd);
    auto sqrt_eigvals = signed_eigvals_backward(s, u, vh).clamp_min(0).sqrt();

    // The upstream formula only uses the symmetric part of the gradient.
    auto grad_sym = 0.5 * (grad + grad.transpose(-2, -1));

    // W_ij = 1 / (sqrt(lambda_i) + sqrt(lambda_j)). The diagonal i == j degenerates to
    // 1 / (2 sqrt(lambda_i)) without a special case, and since the denominator is a sum it does
    // not cancel at repeated eigenvalues.
    auto denom = sqrt_eigvals.unsqueeze(-1) + sqrt_eigvals.unsqueeze(-2);
    auto u_t = u.transpose(-2, -1);
    auto inner = at::matmul(at::matmul(u_t, grad_sym), u).div(denom);
    auto result = at::matmul(at::matmul(u, inner), u_t);
    return 0.5 * (result + result.transpose(-2, -1));
}
#endif
} // namespace op_api
