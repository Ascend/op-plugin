// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t BSND_DIMS = 4;
constexpr int64_t TND_DIMS = 3;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_ONE = 1;

inline void check_mhc_pre_backward_supported()
{
    static const bool is_cann_ready = op_plugin::utils::is_gte_cann_version_900();
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnMhcPre") && check_aclnn_kernel_available("aclnnMhcPreBackward");
    TORCH_CHECK(
        is_cann_ready && is_aclnn_kernel_available,
        "torch_npu.npu_mhc_pre_backward requires CANN >= 9.0.0, aclnnMhcPre and aclnnMhcPreBackward support. "
        "Please upgrade CANN.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mhc_pre_backward(
        const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha,
        const at::Tensor &grad_h_in, const at::Tensor &grad_h_post, const at::Tensor &grad_h_res,
        const at::Tensor &inv_rms, const at::Tensor &h_mix, const at::Tensor &h_pre, const at::Tensor &h_post,
        const c10::optional<at::Tensor> &gamma, double hc_eps, const c10::optional<at::Tensor> &grad_x_post)
{
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.");
    TORCH_CHECK(phi.numel() > 0, "Tensor phi is empty.");
    TORCH_CHECK(alpha.numel() > 0, "Tensor alpha is empty.");
    TORCH_CHECK(grad_h_in.numel()  > 0, "Tensor grad_h_in is empty.");
    TORCH_CHECK(grad_h_post.numel() > 0, "Tensor grad_h_post is empty.");
    TORCH_CHECK(grad_h_res.numel() > 0, "Tensor grad_h_res is empty.");
    TORCH_CHECK(inv_rms.numel() > 0, "Tensor inv_rms is empty.");
    TORCH_CHECK(h_mix.numel() > 0, "Tensor h_mix is empty.");
    TORCH_CHECK(h_pre.numel() > 0, "Tensor h_pre is empty.");
    TORCH_CHECK(h_post.numel() > 0, "Tensor h_post is empty.");
    TORCH_CHECK(x.dim() == TND_DIMS || x.dim() == BSND_DIMS, "Input x must be 3D or 4D, but got ", x.dim(), "D.");
    TORCH_CHECK(grad_h_in.dim() == 2 || grad_h_in.dim() == 3,
      "Input grad_h_in must be 2D or 3D, but got ", grad_h_in.dim(), "D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grad_h_post.dim() == 2 || grad_h_post.dim() == 3,
      "Input grad_h_post must be 2D or 3D, but got ", grad_h_post.dim(), "D.", OPS_ERROR(ErrCode::PARAM));
    check_mhc_pre_backward_supported();

    c10::TensorOptions xGradOptions = x.options().dtype(x.dtype());
    c10::TensorOptions fp32Options = grad_h_in.options().dtype(at::kFloat);

    at::Tensor xGrad;
    at::Tensor phiGrad;
    at::Tensor alphaGrad;
    at::Tensor biasPostGrad;
    at::Tensor gammaGrad;

    auto dimen = grad_h_in.size(grad_h_in.dim() - 1); // [T, D]/[B, S, D]
    auto numsResidual = grad_h_post.size(grad_h_post.dim() - 1); // [T, n]/[B, S, n]

    // 根据x的维度判断shape是BSND格式或TND格式
    if (grad_h_in.dim() == NUM_THREE) {
        auto batch = grad_h_in.size(DIM_0);
        auto sequence = grad_h_in.size(DIM_1);
        c10::SmallVector<int64_t, BSND_DIMS> xGradSize = {batch, sequence, numsResidual, dimen};
        xGrad = at::empty(xGradSize, xGradOptions); // [B, S, n, D]
    } else {
        auto t = grad_h_in.size(DIM_0);
        c10::SmallVector<int64_t, TND_DIMS> xGradSize = {t, numsResidual, dimen};
        xGrad = at::empty(xGradSize, xGradOptions); // [T, n, D]
    }
    c10::SmallVector<int64_t, NUM_TWO> phiGradSize = {(NUM_TWO * numsResidual) + (numsResidual * numsResidual), numsResidual * dimen};
    c10::SmallVector<int64_t, NUM_ONE> alphaGradSize = {NUM_THREE};
    c10::SmallVector<int64_t, NUM_ONE> biasPostGradSize = {(NUM_TWO * numsResidual) + (numsResidual * numsResidual)};
    c10::SmallVector<int64_t, NUM_TWO> gammaGradSize = {numsResidual, dimen};
    phiGrad = at::empty(phiGradSize, fp32Options); // [n^2 + 2n, nD]
    alphaGrad = at::empty(alphaGradSize, fp32Options); // [3]
    biasPostGrad = at::empty(biasPostGradSize, fp32Options); // [n^2 + 2n]
    gammaGrad = at::empty(gammaGradSize, fp32Options); // [n, D]

    float eps_f = static_cast<float>(hc_eps);
    EXEC_NPU_CMD(aclnnMhcPreBackward, x, phi, alpha, grad_h_in, grad_h_post,
            grad_h_res, inv_rms, h_mix, h_pre, h_post, gamma, grad_x_post, eps_f, xGrad, phiGrad, alphaGrad, biasPostGrad, gammaGrad);

    return std::make_tuple(xGrad, phiGrad, alphaGrad, biasPostGrad, gammaGrad);
}

} // namespace op_api
