// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask)
{
    DO_COMPATIBILITY(aclnnGroupNormBackward,
                     acl_op::native_group_norm_backward(dY, X, mean, rstd, gamma_opt, N, C, HxW, group,
                                                        grad_input_mask));
    at::Tensor grad_x;
    at::Tensor grad_gamma;
    at::Tensor grad_beta;
    if (grad_input_mask[0]) {
        grad_x = npu_preparation::apply_tensor_without_format(X.sizes(), X.options().dtype(at::kFloat));
    }
    if (grad_input_mask[1]) {
        grad_gamma = npu_preparation::apply_tensor_without_format({C}, X.options().dtype(at::kFloat));
        grad_gamma.zero_();
    }
    if (grad_input_mask[2]) {
        grad_beta = npu_preparation::apply_tensor_without_format({C}, X.options().dtype(at::kFloat));
        grad_beta.zero_();
    }
    EXEC_NPU_CMD(aclnnGroupNormGrad, dY, mean, rstd, X, gamma_opt,
                 group, "NCHW", grad_input_mask[0], grad_input_mask[1], grad_input_mask[2],
                 grad_x, grad_gamma, grad_beta);
    auto result_dtype = dY.scalar_type();
    at::Tensor grad_x_cp;
    at::Tensor grad_gamma_cp;
    at::Tensor grad_beta_cp;
    if (grad_input_mask[0]) {
        grad_x_cp = grad_x.scalar_type() == result_dtype ? grad_x :
            at_npu::native::custom_ops::npu_dtype_cast(grad_x, result_dtype);
    }
    if (grad_input_mask[1]) {
        grad_gamma_cp = grad_x.scalar_type() == result_dtype ? grad_gamma :
            at_npu::native::custom_ops::npu_dtype_cast(grad_gamma, result_dtype);
    }
    if (grad_input_mask[2]) {
        grad_beta_cp = grad_x.scalar_type() == result_dtype ? grad_beta :
            at_npu::native::custom_ops::npu_dtype_cast(grad_beta, result_dtype);
    }
    return std::make_tuple(grad_x_cp, grad_gamma_cp, grad_beta_cp);
}

} // namespace op_api
