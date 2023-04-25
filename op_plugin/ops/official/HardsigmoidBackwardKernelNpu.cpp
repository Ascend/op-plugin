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

at::Tensor hardsigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  at::Tensor grad_input = npu_preparation::ApplyTensor(grad_output);
  at_npu::native::OpCommand cmd;
  cmd.Name("HardSigmoidGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Run();

  return grad_input;
}
} // namespace op_plugin
