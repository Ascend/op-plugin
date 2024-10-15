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

at::Tensor npu_rotary_mul(
    const at::Tensor& self,
    const at::Tensor& cos,
    const at::Tensor& sin)
{
    DO_COMPATIBILITY(aclnnRotaryPositionEmbedding, acl_op::npu_rotary_mul(self, cos, sin));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::npu_rotary_mul(self, cos, sin);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options());
    int64_t mode = 0;
    EXEC_NPU_CMD(aclnnRotaryPositionEmbedding, self, cos, sin, mode, result);
    return result;
}
}
