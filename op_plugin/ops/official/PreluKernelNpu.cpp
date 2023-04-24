// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

at::Tensor prelu(const at::Tensor& self, const at::Tensor& weight) {
  at::Tensor result = npu_preparation::ApplyTensor(self);

  at_npu::native::OpCommand cmd;
  cmd.Name("PRelu")
      .Input(self)
      .Input(weight)
      .Output(result)
      .Run();
  return result;
}

at::Tensor _prelu_kernel(const at::Tensor& self, const at::Tensor& weight) {
  return op_plugin::prelu(self, weight);
}
} // namespace op_plugin
