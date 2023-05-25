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
namespace {
at::Tensor& gelu_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self) {
  at::Tensor unused = grad;
  at_npu::native::OpCommand cmd;
  cmd.Name("GeluGrad")
      .Input(grad)
      .Input(self)
      .Input(unused)
      .Output(grad_input)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor gelu_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    c10::string_view approximate) {
  at::Tensor grad_input = at_npu::native::OpPreparation::apply_tensor(self);
  gelu_backward_out_nocheck(grad_input, grad, self);
  return grad_input;
}
} // namespace op_plugin
