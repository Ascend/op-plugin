// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_fast_gelu(const at::Tensor &self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                            c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                            (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return acl_op::npu_fast_gelu(self);
    }

    DO_COMPATIBILITY(aclnnFastGelu, acl_op::npu_fast_gelu(self));

    at::Tensor result = npu_preparation::apply_tensor(self);
    EXEC_NPU_CMD(aclnnFastGelu, self, result);
    return result;
}

at::Tensor npu_fast_gelu_backward(const at::Tensor &grad, const at::Tensor &self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                            c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                            (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return acl_op::npu_fast_gelu_backward(grad, self);
    }

    DO_COMPATIBILITY(aclnnFastGeluBackward, acl_op::npu_fast_gelu_backward(grad, self));

    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    EXEC_NPU_CMD(aclnnFastGeluBackward, grad, self, grad_input);
    return grad_input;
}

at::Tensor fast_gelu(const at::Tensor &self)
{
    return op_api::npu_fast_gelu(self);
}

} // namespace op_api