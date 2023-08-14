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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
at::Tensor col2im_backward(
    const at::Tensor& self, 
    at::IntArrayRef kernel_size, 
    at::IntArrayRef dilation,
    at::IntArrayRef padding, 
    at::IntArrayRef stride) {
  return op_plugin::im2col(self, kernel_size, dilation, padding, stride);
}

at::Tensor& col2im_backward_out(
    const at::Tensor& self, 
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding, 
    at::IntArrayRef stride, 
    at::Tensor& result) {
  return op_plugin::im2col_out(self, kernel_size, dilation, padding, stride, result);
}
} // namespace op_plugin
