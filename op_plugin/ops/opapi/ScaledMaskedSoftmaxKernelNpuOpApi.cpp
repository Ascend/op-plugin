// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at related link.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/AccumulateType.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_scaled_masked_softmax(const at::Tensor& x, const at::Tensor& mask, const at::Scalar& scale,
    bool fixed_triu_mask)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        return acl_op::npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask);
    }
    DO_COMPATIBILITY(aclnnScaledMaskedSoftmax, acl_op::npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask));
    double scale_value = 1.0;
    if (scale.isFloatingPoint()) {
        scale_value = scale.to<double>();
    } else if (scale.isIntegral(true)) {
        scale_value = static_cast<double>(scale.to<int64_t>());
    } else {
        TORCH_CHECK(false, "scaled_masked_softmax expects scale to be float or int", OPS_ERROR(ErrCode::TYPE));
    }

    at::Tensor result = npu_preparation::apply_tensor_without_format(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnScaledMaskedSoftmax, x, mask, scale_value, fixed_triu_mask, result);
    return result;
}

at::Tensor npu_scaled_masked_softmax_backward(
    const at::Tensor& y_grad,
    const at::Tensor& y,
    const at::Tensor& mask,
    const at::Scalar& scale,
    bool fixed_triu_mask)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        return acl_op::npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask);
    }
    DO_COMPATIBILITY(aclnnScaledMaskedSoftmaxBackward, acl_op::npu_scaled_masked_softmax_backward(y_grad, y, mask, scale, fixed_triu_mask));
    double scale_value = 1.0;
    if (scale.isFloatingPoint()) {
        scale_value = scale.to<double>();
    } else if (scale.isIntegral(true)) {
        scale_value = static_cast<double>(scale.to<int64_t>());
    } else {
        TORCH_CHECK(false, "scaled_masked_softmax_backward expects scale to be float or int", OPS_ERROR(ErrCode::TYPE));
    }

    at::Tensor result = npu_preparation::apply_tensor_without_format(y_grad.sizes(), y_grad.options());
    EXEC_NPU_CMD(aclnnScaledMaskedSoftmaxBackward, y_grad, y, mask, scale_value, fixed_triu_mask, result);
    return result;
}
}