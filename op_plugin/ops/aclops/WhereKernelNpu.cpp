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

namespace {
at::SmallVector<int64_t, SIZE> where_npu_output_size(const at::Tensor& condition) {
  int64_t dim = condition.dim();
  at::Tensor boolSelf = at_npu::native::custom_ops::npu_dtype_cast(condition, at::ScalarType::Bool);
  at::Tensor intSelf = at_npu::native::custom_ops::npu_dtype_cast(boolSelf, at::ScalarType::Int);
  at::Tensor cout_nonzero_self = at::sum(intSelf, at::ScalarType::Int);
  int64_t nonzero_num = cout_nonzero_self.item().toInt();
  at::SmallVector<int64_t, SIZE> output_size = {nonzero_num, dim};
  return output_size;
}
} // namespace

std::vector<at::Tensor> where(const at::Tensor& condition) {
  at::Tensor format_cast_of_condition = condition;
  if (npu_preparation::get_tensor_npu_format(condition) != ACL_FORMAT_ND) {
    format_cast_of_condition =
        at_npu::native::custom_ops::npu_format_cast(format_cast_of_condition, ACL_FORMAT_ND);
  }
  if (condition.scalar_type() == at::ScalarType::Half) {
    format_cast_of_condition = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_condition, at::ScalarType::Float);
  }

  auto output_size = where_npu_output_size(format_cast_of_condition);
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      output_size, format_cast_of_condition.options().dtype(at::kLong), ACL_FORMAT_ND);

  at_npu::native::OpCommand cmd;
  cmd.Name("NonZero")
      .Input(format_cast_of_condition)
      .Output(result)
      .Run();
  result = result.transpose(1, 0);
  std::vector<at::Tensor> chunk_result = result.chunk(result.size(0), 0);
  std::vector<at::Tensor> squeeze_result;
  for (uint64_t i = 0; i < chunk_result.size(); i++) {
    squeeze_result.push_back(chunk_result[i].squeeze(0));
  }

  return squeeze_result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor _s_where(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    at::Tensor result = npu_preparation::apply_tensor(self);

    at_npu::native::OpCommand cmd;
    cmd.Name("Select").Input(condition).Input(self).Input(other).Output(result).Run();

    return result;
}

at::Tensor where(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
                "expected condition, x and y to be on the same device, but condition is on ", condition.device(),
                " and x and y are on ", self.device(), " and ", other.device(), " respectively", OPS_ERROR(ErrCode::PARAM));
    if (condition.scalar_type() != at::ScalarType::Byte && condition.scalar_type() != at::ScalarType::Bool) {
        AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ", toString(condition.scalar_type()));
    }
    at::Tensor b_condition;
    at::Tensor b_self;
    at::Tensor b_other;
    std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
    return at::_s_where(b_condition, b_self, b_other);
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor &where_out(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    at::Tensor b_condition;
    at::Tensor b_self;
    at::Tensor b_other;
    std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
    if (self.dtype() != other.dtype()) {
        auto result_type = at::native::result_type(self, other);
        b_self = at_npu::native::custom_ops::npu_dtype_cast(b_self, result_type);
        b_other = at_npu::native::custom_ops::npu_dtype_cast(b_other, result_type);
    }
    npu_preparation::CheckOut({condition, self, other}, out, b_self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out);
        where_out_nocheck(contiguous_out, condition, self, other);
        npu_utils::format_fresh_view(out, contiguous_out);
    } else {
        where_out_nocheck(out, condition, self, other);
    }

    return out;
}

at::Tensor where(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    at::Tensor b_condition, b_self, b_other;
    std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
    at::Tensor ret = npu_preparation::apply_tensor(b_self);
    where_out_nocheck(ret, b_condition, b_self, b_other);
    return ret;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor& where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    at::Tensor b_condition;
    at::Tensor b_self;
    at::Tensor b_other;
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
    at::Tensor b_condition;
    at::Tensor b_self;
    at::Tensor b_other;
    std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
    at::Tensor ret = npu_preparation::apply_tensor(b_self);
    where_out_nocheck(ret, b_condition, b_self, b_other);
    return ret;
}
#endif
} // namespace acl_op
