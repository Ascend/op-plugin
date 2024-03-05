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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_normalize_batch(
    const at::Tensor& self,
    const at::Tensor& seq_len,
    int64_t normalize_type) {
  TORCH_CHECK(
      seq_len.dim() == 1,
      "Non-empty 1D seq_len tensor expected but got a tensor with sizes ",
      seq_len.sizes(),
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      seq_len.size(0) == self.size(0),
      "seq_len's length should be equal self' num, but got seq_len length ",
      seq_len.size(0),
      "self num ",
      self.size(0),
      OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      normalize_type >= 0 && normalize_type <= 1,
      "normalize_type expected to be in range [0, 1], but got ",
      normalize_type,
      OPS_ERROR(ErrCode::VALUE));

  at::Tensor result = npu_preparation::apply_tensor(self);
  string normalize_type_str = normalize_type == 0 ? "per_feature" : "all_features";

  constexpr float_t EPSILON = 1e-5;
  at_npu::native::OpCommand cmd;
  cmd.Name("NormalizeBatch")
      .Input(self)
      .Input(seq_len)
      .Output(result)
      .Attr("normalize_type", normalize_type_str)
      .Attr("epsilon", EPSILON)
      .Run();
  return result;
}
} // namespace acl_op
