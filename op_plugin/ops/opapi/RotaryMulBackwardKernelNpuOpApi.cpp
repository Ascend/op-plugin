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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& cos,
    const at::Tensor& sin)
{
    DO_COMPATIBILITY(aclnnRotaryPositionEmbeddingGrad, acl_op::npu_rotary_mul_backward(grad, self, cos, sin));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::npu_rotary_mul_backward(grad, self, cos, sin);
    }
    at::Tensor dx = npu_preparation::apply_tensor_without_format(grad.sizes(), grad.options());
    at::Tensor dcos = npu_preparation::apply_tensor_without_format(cos.sizes(), cos.options());
    at::Tensor dsin = npu_preparation::apply_tensor_without_format(sin.sizes(), sin.options());
    int64_t mode = 0;
    EXEC_NPU_CMD(aclnnRotaryPositionEmbeddingGrad, grad, cos, sin, self, mode, dx, dcos, dsin);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(dx, dcos, dsin);
}
}
