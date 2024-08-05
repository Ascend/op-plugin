// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& multinomial_op_api(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const uint64_t seed = pair.first;
  const uint64_t offset = pair.second;

  EXEC_NPU_CMD(aclnnMultinomial, self, num_samples, replacement, seed, offset, result);
  return result;
}

at::Tensor& multinomial_out(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMultinomial, acl_op::multinomial_out(self, num_samples, replacement, gen, result));
  auto input_dim = self.dim();
  auto output_size = op_infer::array_to_small_vector(self.sizes());
  output_size[input_dim - 1] = num_samples;
  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      at::ScalarType::Long,
      output_size);
  multinomial_op_api(result, self, num_samples, replacement, gen);
  return result;
}

at::Tensor multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnMultinomial, acl_op::multinomial(self, num_samples, replacement, gen));
  auto dim = self.dim();
  auto shape = op_infer::array_to_small_vector(self.sizes());
  shape[dim-1] = num_samples;
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(
                      shape, self.options().dtype(at::kLong));
  multinomial_op_api(result, self, num_samples, replacement, gen);
  return result;
}

}
