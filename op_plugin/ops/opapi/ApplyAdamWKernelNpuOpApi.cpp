// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace c10_npu;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam_w(
    const at::Scalar &beta1_power,
    const at::Scalar &beta2_power,
    const at::Scalar &lr,
    const at::Scalar &weight_decay,
    const at::Scalar &beta1,
    const at::Scalar &beta2,
    const at::Scalar &epsilon,
    const at::Tensor &grad,
    const c10::optional<at::Tensor> &max_grad_norm,
    c10::optional<bool> amsgrad,
    c10::optional<bool> maximize)
{
    TORCH_CHECK(false, "npu_apply_adam_w is not implemented for Tensor"
       + OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_w_out(
    const at::Scalar &beta1_power,
    const at::Scalar &beta2_power,
    const at::Scalar &lr,
    const at::Scalar &weight_decay,
    const at::Scalar &beta1,
    const at::Scalar &beta2,
    const at::Scalar &epsilon,
    const at::Tensor &grad,
    const c10::optional<at::Tensor> &max_grad_norm,
    c10::optional<bool> amsgrad,
    c10::optional<bool> maximize,
    at::Tensor &var,
    at::Tensor &m,
    at::Tensor &v)
{
    DO_COMPATIBILITY(aclnnApplyAdamW, acl_op::npu_apply_adam_w_out(beta1_power, beta2_power, lr,
        weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v));

    if ((amsgrad.has_value()) && (amsgrad.value())) {
        TORCH_CHECK(max_grad_norm.has_value(),"if amsgrad is true, max_gard_norm input must be entered"
            + OPS_ERROR(ErrCode::PARAM));
    }

    if (c10_npu::IsAclnnOnly()) {
        at::Tensor beta1_power_tensor = npu_preparation::copy_scalar_to_device(beta1_power, grad.scalar_type(), grad.device());
        at::Tensor beta2_power_tensor = npu_preparation::copy_scalar_to_device(beta2_power, grad.scalar_type(), grad.device());
        at::Tensor lr_tensor = npu_preparation::copy_scalar_to_device(lr, grad.scalar_type(), grad.device());
        at::Tensor weight_decay_tensor = npu_preparation::copy_scalar_to_device(weight_decay, grad.scalar_type(), grad.device());
        at::Tensor beta1_tensor = npu_preparation::copy_scalar_to_device(beta1, grad.scalar_type(), grad.device());
        at::Tensor beta2_tensor = npu_preparation::copy_scalar_to_device(beta2, grad.scalar_type(), grad.device());
        at::Tensor epsilon_tensor = npu_preparation::copy_scalar_to_device(epsilon, grad.scalar_type(), grad.device());

        bool amsgrad_value = amsgrad.value_or(false);
        bool maximize_value = maximize.value_or(false);
        EXEC_NPU_CMD(aclnnApplyAdamW, var, m, v, beta1_power_tensor, beta2_power_tensor, lr_tensor,
            weight_decay_tensor, beta1_tensor, beta2_tensor, epsilon_tensor, grad,
            max_grad_norm, amsgrad_value, maximize_value);
        return std::tie(var, m, v);
    } else {
        TORCH_NPU_WARN("current soc not support aclnn");
        return acl_op::npu_apply_adam_w_out(beta1_power, beta2_power, lr,
            weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
    }
}
}
