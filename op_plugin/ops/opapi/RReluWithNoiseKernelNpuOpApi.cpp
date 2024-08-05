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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor rrelu_with_noise(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator)
{
  DO_COMPATIBILITY(aclnnRReluWithNoise, acl_op::rrelu_with_noise(self, noise, lower, upper, training, generator));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  auto gen_ =
    at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(1 << 28);
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);

  EXEC_NPU_CMD(aclnnRReluWithNoise, self, noise, lower, upper, training, seed, offset, result);

  return result;
}

at::Tensor& rrelu_with_noise_(
    at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator)
{
  DO_COMPATIBILITY(aclnnInplaceRReluWithNoise,
                   acl_op::rrelu_with_noise_(self, noise, lower, upper, training, generator));
  auto gen_ =
    at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(1 << 28);
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  EXEC_NPU_CMD(aclnnInplaceRReluWithNoise, self, noise, lower, upper, training, seed, offset);

  return self;
}

at::Tensor& rrelu_with_noise_out(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    c10::optional<at::Generator> generator,
    at::Tensor& output)
{
  DO_COMPATIBILITY(aclnnRReluWithNoise,
                   acl_op::rrelu_with_noise_out(self, noise, lower, upper, training, generator, output));
  npu_preparation::check_tensor(
      {self, noise},
      output,
      self);
  auto gen_ =
    at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(1 << 28);
  const int64_t seed = static_cast<int64_t>(pair.first);
  const int64_t offset = static_cast<int64_t>(pair.second);
  EXEC_NPU_CMD(aclnnRReluWithNoise, self, noise, lower, upper, training, seed, offset, output);

  return output;
}

}
