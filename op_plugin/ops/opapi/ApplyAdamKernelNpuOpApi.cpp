// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam(
    const at::Scalar &beta1_power,
    const at::Scalar &beta2_power,
    const at::Scalar &lr,
    const at::Scalar &beta1,
    const at::Scalar &beta2,
    const at::Scalar &epsilon,
    const at::Tensor &grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov)
{
    TORCH_CHECK(false, "npu_apply_adam is not implemented for Tensor"
       + OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_out(
    const at::Scalar &beta1_power,
    const at::Scalar &beta2_power,
    const at::Scalar &lr,
    const at::Scalar &beta1,
    const at::Scalar &beta2,
    const at::Scalar &epsilon,
    const at::Tensor &grad,
    c10::optional<bool> use_locking,
    c10::optional<bool> use_nesterov,
    at::Tensor &var,
    at::Tensor &m,
    at::Tensor &v)
{
    DO_COMPATIBILITY(aclnnApplyAdam, acl_op::npu_apply_adam_out(beta1_power, beta2_power, lr,
        beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v));

    if (c10_npu::IsAclnnOnly()) {
        at::Tensor beta1_power_tensor = npu_preparation::copy_scalar_to_device(beta1_power, at::kFloat, grad.device());
        at::Tensor beta2_power_tensor = npu_preparation::copy_scalar_to_device(beta2_power, at::kFloat, grad.device());
        at::Tensor lr_tensor = npu_preparation::copy_scalar_to_device(lr, at::kFloat, grad.device());
        at::Tensor beta1_tensor = npu_preparation::copy_scalar_to_device(beta1, at::kFloat, grad.device());
        at::Tensor beta2_tensor = npu_preparation::copy_scalar_to_device(beta2, at::kFloat, grad.device());
        at::Tensor epsilon_tensor = npu_preparation::copy_scalar_to_device(epsilon, at::kFloat, grad.device());

        bool use_locking_value = use_locking.value_or(false);
        bool use_nesterov_value = use_nesterov.value_or(false);
        EXEC_NPU_CMD(aclnnApplyAdam, var, m, v, beta1_power_tensor, beta2_power_tensor, lr_tensor,
            beta1_tensor, beta2_tensor, epsilon_tensor, grad, use_locking_value, use_nesterov_value);
        return std::tie(var, m, v);
    } else {
        TORCH_NPU_WARN("current soc not support aclnn");
        return acl_op::npu_apply_adam_out(beta1_power, beta2_power, lr,
            beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
    }
}
}
