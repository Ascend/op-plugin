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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
at::Tensor& im2col_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& grad_input) {
  return acl_op::col2im_out(grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);
}

at::Tensor im2col_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  return acl_op::col2im(grad_output, input_size, kernel_size, dilation, padding, stride);
}
} // namespace acl_op
