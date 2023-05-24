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
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::tuple<at::Tensor, at::Tensor> prelu_backward_out_nocheck(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight) {
  at_npu::native::OpCommand cmd;
  cmd.Name("PReluGrad")
      .Input(grad_output)
      .Input(self)
      .Input(weight)
      .Output(grad_input)
      .Output(grad_weight)
      .Run();

  return std::tuple<at::Tensor, at::Tensor>(grad_input, grad_weight);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _prelu_kernel_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight) {
  at::Tensor grad_input = npu_preparation::ApplyTensor(self);
  at::Tensor grad_weight = npu_preparation::ApplyTensor(weight);
  prelu_backward_out_nocheck(grad_input, grad_weight, grad_output, self, weight);

  return std::tie<at::Tensor, at::Tensor>(grad_input, grad_weight);
}
} // namespace op_plugin
