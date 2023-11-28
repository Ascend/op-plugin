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
        grad_x = npu_preparation::apply_tensor_without_format(dY);
    }
    if (grad_input_mask[1]) {
        grad_gamma = npu_preparation::apply_tensor_without_format(X, {C});
    }
    if (grad_input_mask[2]) {
        grad_beta = npu_preparation::apply_tensor_without_format(X, {C});
    }

    EXEC_NPU_CMD(aclnnGroupNormBackward, dY, X, mean, rstd, gamma_opt, N, C, HxW, group, grad_input_mask,
                 grad_x, grad_gamma, grad_beta);
    return std::make_tuple(grad_x, grad_gamma, grad_beta);
}

} // namespace op_api
