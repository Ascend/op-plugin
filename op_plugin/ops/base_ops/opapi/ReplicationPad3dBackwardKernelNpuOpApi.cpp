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

at::Tensor& replication_pad3d_backward_out(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           at::IntArrayRef padding,
                                           at::Tensor& grad_input)
{
    npu_preparation::check_tensor({self, grad_output}, grad_input, self);
    EXEC_NPU_CMD(aclnnReplicationPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor replication_pad3d_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      at::IntArrayRef padding)
{
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnReplicationPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

}  // namespace op_api

