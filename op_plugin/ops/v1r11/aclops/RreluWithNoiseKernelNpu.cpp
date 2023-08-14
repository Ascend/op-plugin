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

#include <ATen/core/DistributionsHelper.h>

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
void _rrelu_with_noise_train(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& noise,
    at::Scalar lower_,
    at::Scalar upper_,
    c10::optional<at::Generator> generator) {
  // use vector calculation instead of point-loop calculation
  double lower = lower_.toDouble();
  double upper = upper_.toDouble();
  at::Tensor uniform_tensor = at::empty(input.sizes(), input.options()).uniform_(lower, upper, generator);
  at::Tensor mask_tensor = input.le(0);
  at::Tensor one_tensor = at::empty(input.sizes(), input.options()).fill_(1).to(noise.dtype());
  at::Tensor select_tensor = at::_s_where(mask_tensor, uniform_tensor, one_tensor);
  noise.copy_(select_tensor);
  at::Tensor result = output.contiguous();
  result = input.mul(noise);
  output.copy_(result);
}

at::Tensor& rrelu_with_noise_out_nocheck(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator) {
  if (training) {
    _rrelu_with_noise_train(output, self.contiguous(), noise, lower, upper, generator);
    return output;
  } else {
    auto float_lower = lower.toFloat();
    auto float_upper = upper.toFloat();
    at::Scalar negative_slope = (float_lower + float_upper) / 2;
    return op_plugin::leaky_relu_out(self, negative_slope, output);
  }
}
} // namespace

at::Tensor rrelu_with_noise(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return rrelu_with_noise_out_nocheck(output, self, noise, lower, upper, training, generator);
}

at::Tensor& rrelu_with_noise_(
    at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator) {
  return op_plugin::rrelu_with_noise_out(self, noise, lower, upper, training, generator, self);
}

at::Tensor& rrelu_with_noise_out(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator,
    at::Tensor& output) {
  npu_preparation::CheckOut(
      {self, noise},
      output,
      self);

  if (!npu_utils::check_match(&output)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(output);
    rrelu_with_noise_out_nocheck(contiguous_result, self, noise, lower, upper, training, generator);
    npu_utils::format_fresh_view(output, contiguous_result);
  } else {
    rrelu_with_noise_out_nocheck(output, self, noise, lower, upper, training, generator);
  }

  return output;
}
} // namespace op_plugin
