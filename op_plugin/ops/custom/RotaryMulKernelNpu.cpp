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
at::Tensor& rotary_mul_nocheck(
    at::Tensor& y,
    const at::Tensor& x,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  at_npu::native::OpCommand cmd;
  cmd.Name("RotaryMul")
      .Input(x)
      .Input(r1)
      .Input(r2)
      .Output(y)
      .Run();
  return y;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> rotary_mul_backward_nocheck(
    at::Tensor& dx,
    at::Tensor& dr1,
    at::Tensor& dr2,
    const at::Tensor& x,
    const at::Tensor& r1,
    const at::Tensor& r2,
    const at::Tensor& dy) {
  at_npu::native::OpCommand cmd;
  cmd.Name("RotaryMulGrad")
      .Input(x)
      .Input(r1)
      .Input(r2)
      .Input(dy)
      .Output(dx)
      .Output(dr1)
      .Output(dr2)
      .Run();
  return std::tie(dx, dr1, dr2);
}
} // namespace

at::Tensor npu_rotary_mul(
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  rotary_mul_nocheck(result, self, r1, r2);
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2) {
  at::Tensor dx = npu_preparation::ApplyTensor(self);
  at::Tensor dr1 = npu_preparation::ApplyTensor(r1);
  at::Tensor dr2 = npu_preparation::ApplyTensor(r2);
  rotary_mul_backward_nocheck(dx, dr1, dr2, self, r1, r2, grad);
  return std::tie(dx, dr1, dr2);
}
} // namespace op_plugin
