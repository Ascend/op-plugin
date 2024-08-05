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

#include <ATen/core/DistributionsHelper.h>

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
#if VERSION_BETWEEN(V1R11, V1R11)
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
    return acl_op::leaky_relu_out(self, negative_slope, output);
  }
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
void _rrelu_with_noise_train(at::Tensor &output, const at::Tensor &input, const at::Tensor &noise, at::Scalar lower_,
                             at::Scalar upper_, c10::optional<at::Generator> generator)
{
    float lower = lower_.toFloat();
    float upper = upper_.toFloat();
    auto shape = output.sizes();
    auto noise_shape = noise.sizes();
    at::Tensor tmp_tensor = output.contiguous();
    at::Tensor output_data = tmp_tensor.reshape({output.numel()});
    at::Tensor input_data = input.reshape({input.numel()});
    at::Tensor tmp_noise = noise;
    tmp_noise = tmp_noise.reshape({tmp_noise.numel()});
    auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(generator, at::detail::getDefaultCPUGenerator());

    for (int64_t i = 0; i < input.numel(); i++) {
        if (input_data[i].item().toFloat() <= 0) {
            at::uniform_real_distribution<double> uniform(lower, upper);
            const float r = uniform(gen);
            output_data[i] = input_data[i] * r;
            tmp_noise[i] = r;
        } else {
            tmp_noise[i] = 1;
            output_data[i] = input_data[i];
        }
    }

    if (!output.is_contiguous()) {
        output.copy_(tmp_tensor);
    }
    tmp_noise.reshape(noise_shape);
    noise.copy_(tmp_noise);
    output.reshape(shape);
}

at::Tensor &rrelu_with_noise_out_nocheck(at::Tensor &output, const at::Tensor &self, const at::Tensor &noise,
                                         const at::Scalar &lower, const at::Scalar &upper, bool training,
                                         c10::optional<at::Generator> generator)
{
    if (training) {
        _rrelu_with_noise_train(output, self.contiguous(), noise, lower, upper, generator);
        return output;
    } else {
        auto float_lower = lower.toFloat();
        auto float_upper = upper.toFloat();
        at::Scalar negative_slope = (float_lower + float_upper) / 2;
        return acl_op::leaky_relu_out(self, negative_slope, output);
    }
}
#endif
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
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
  return acl_op::rrelu_with_noise_out(self, noise, lower, upper, training, generator, self);
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
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor rrelu_with_noise(const at::Tensor &self, const at::Tensor &noise, const at::Scalar &lower,
                            const at::Scalar &upper, bool training, c10::optional<at::Generator> generator)
{
    TORCH_CHECK(noise.sizes().equals(self.sizes()), "The shape of noise must equal to the shape of self!" + OPS_ERROR(ErrCode::PARAM));
    auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return rrelu_with_noise_out_nocheck(output, self, noise, lower, upper, training, generator);
}

at::Tensor &rrelu_with_noise_(at::Tensor &self, const at::Tensor &noise, const at::Scalar &lower,
                              const at::Scalar &upper, bool training, c10::optional<at::Generator> generator)
{
    TORCH_CHECK(noise.sizes().equals(self.sizes()), "The shape of noise must equal to the shape of self!" + OPS_ERROR(ErrCode::PARAM));
    return acl_op::rrelu_with_noise_out(self, noise, lower, upper, training, generator, self);
}

at::Tensor &rrelu_with_noise_out(const at::Tensor &self, const at::Tensor &noise, const at::Scalar &lower,
                                 const at::Scalar &upper, bool training, c10::optional<at::Generator> generator,
                                 at::Tensor &output)
{
    TORCH_CHECK(noise.sizes().equals(self.sizes()), "The shape of noise must equal to the shape of self!" + OPS_ERROR(ErrCode::PARAM));
    npu_preparation::CheckOut({self, noise}, output, self);

    if (!npu_utils::check_match(&output)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(output);
        rrelu_with_noise_out_nocheck(contiguous_result, self, noise, lower, upper, training, generator);
        npu_utils::format_fresh_view(output, contiguous_result);
    } else {
        rrelu_with_noise_out_nocheck(output, self, noise, lower, upper, training, generator);
    }

    return output;
}
#endif

} // namespace acl_op
