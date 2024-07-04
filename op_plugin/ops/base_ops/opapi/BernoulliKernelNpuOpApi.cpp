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

#include <climits>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
static const uint64_t PHILOX_DEFAULT_NUM = 10;

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulli, acl_op::bernoulli_(self, p, gen));
  auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(PHILOX_DEFAULT_NUM);
  const uint64_t seed = pair.first;
  const uint64_t offset = pair.second;

  const c10::Scalar& pScalar = at::Scalar(p);
  EXEC_NPU_CMD(aclnnInplaceBernoulli, self, pScalar, seed, offset);
  return self;
}

at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, acl_op::bernoulli_(self, p, gen));
  auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(PHILOX_DEFAULT_NUM);
  const uint64_t seed = pair.first;
  const uint64_t offset = pair.second;
  EXEC_NPU_CMD(aclnnInplaceBernoulliTensor, self, p, seed, offset);
  return self;
}

at::Tensor bernoulli(const at::Tensor& self, c10::optional<at::Generator> gen) {
    DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, acl_op::bernoulli(self, gen));
    at::Tensor self_copy = npu_preparation::apply_tensor_without_format(self);
    at::namedinference::propagate_names(self_copy, self);
    return op_api::bernoulli_(self_copy, self, gen);
}

at::Tensor bernoulli(const at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulli, acl_op::bernoulli(self, p, gen));
  return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT).bernoulli_(p, gen);
}

at::Tensor& bernoulli_out(const at::Tensor& self, c10::optional<at::Generator> gen, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, acl_op::bernoulli_out(self, gen, result));
  result.resize_(self.sizes()).bernoulli_(self, gen);
  at::namedinference::propagate_names(result, self);
  return result;
}
}  // namespace op_api
