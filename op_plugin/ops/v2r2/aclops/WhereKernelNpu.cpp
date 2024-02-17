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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  at::Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
  npu_preparation::CheckOut(
      {condition, self, other},
      out,
      b_self);
  if (!npu_utils::check_match(&out)) {
    at::Tensor contiguous_out = npu_utils::format_contiguous(out);
    where_out_nocheck(contiguous_out, condition, self, other);
    npu_utils::format_fresh_view(out, contiguous_out);
  } else {
    where_out_nocheck(out, condition, self, other);
  }

  return out;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
  at::Tensor ret = npu_preparation::apply_tensor(b_self);
  where_out_nocheck(ret, b_condition, b_self, b_other);
  return ret;
}

} // namespace acl_op
