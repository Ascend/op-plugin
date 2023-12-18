// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

at::Tensor max_pool2d_with_indices_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices)
{
    DO_COMPATIBILITY(aclnnMaxPool2dWithMaskBackward,
                     acl_op::max_pool2d_with_indices_backward(grad_output, self, kernel_size,
                                                              stride, padding, dilation, ceil_mode, indices));
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(self);

    EXEC_NPU_CMD(aclnnMaxPool2dWithMaskBackward, grad_output, self, indices, kernel_size,
                 stride, padding, dilation, ceil_mode, grad_input);
    return grad_input;
}

at::Tensor& max_pool2d_with_indices_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    DO_COMPATIBILITY(aclnnMaxPool2dWithMaskBackward,
                     acl_op::max_pool2d_with_indices_backward_out(grad_output, self, kernel_size, stride, padding,
                                                                  dilation, ceil_mode, indices, grad_input));

    EXEC_NPU_CMD(aclnnMaxPool2dWithMaskBackward, grad_output, self, indices, kernel_size,
                 stride, padding, dilation, ceil_mode, grad_input);
    return grad_input;
}

} // namespace op_api
