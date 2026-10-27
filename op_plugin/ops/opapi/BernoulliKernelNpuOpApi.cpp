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
#include "op_plugin/utils/RandomUtil.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"

namespace op_api {
static const uint64_t PHILOX_DEFAULT_NUM = 10;

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  at_npu::native::assert_no_internal_overlap(self);
  if (self.numel() == 0) {
    return self; // 对齐社区：空 tensor 提前返回，不消耗 generator offset
  }
  DO_COMPATIBILITY(aclnnInplaceBernoulli, acl_op::bernoulli_(self, p, gen));
  auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  // A5(Ascend950+) 对齐社区：scalar 版 offset 按 numel 动态计算；其他芯片保持固定 10
  const uint64_t counter_offset = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950)
      ? static_cast<uint64_t>(op_plugin::utils::calc_final_counter_offset(self))
      : PHILOX_DEFAULT_NUM;
  auto pair = gen_->philox_engine_inputs(counter_offset);
  const uint64_t seed = pair.first;
  const uint64_t offset = pair.second;

  const c10::Scalar& pScalar = at::Scalar(p);
  EXEC_NPU_CMD(aclnnInplaceBernoulli, self, pScalar, seed, offset);
  return self;
}

at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p, c10::optional<at::Generator> gen) {
  at_npu::native::assert_no_internal_overlap(self);
  if (self.numel() == 0) {
    return self; // 对齐社区：空 tensor 提前返回，不消耗 generator offset
  }
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, acl_op::bernoulli_(self, p, gen));
  at::Tensor cp_p = p;
  if (npu_preparation::IsCPUScalar(p)) {
    at::Scalar scalar = p.item();
    cp_p = npu_preparation::copy_scalar_to_device(scalar, p.scalar_type(), self.device());
  }
  auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(PHILOX_DEFAULT_NUM);
  const uint64_t seed = pair.first;
  const uint64_t offset = pair.second;
  EXEC_NPU_CMD(aclnnInplaceBernoulliTensor, self, cp_p, seed, offset);
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
} // namespace op_api
