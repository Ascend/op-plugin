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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_expand_outplace(
    const at::Tensor& to_expand1,
    const at::Tensor& to_expand2,
    const at::Tensor& to_expand3,
    const char *api_name) {
  for (auto& t : {to_expand1, to_expand2, to_expand3}) {
    if (!t.defined()) {
      TORCH_CHECK(false, api_name, "(...) called with an undefined Tensor");
    }
  }

  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = op_infer::broadcast_ops_npu_output_size(to_expand1, to_expand2);
  auto expanded_size = op_infer::broadcast_ops_npu_output_size(expanded_size12, to_expand3.sizes());

  return std::make_tuple(
      to_expand1.expand(expanded_size, true),
      to_expand2.expand(expanded_size, true),
      to_expand3.expand(expanded_size, true));
}

at::SmallVector<int64_t, SIZE> where_npu_output_size(const at::Tensor& condition){
  int64_t dim = condition.dim();
  at::Tensor boolSelf = op_plugin::npu_dtype_cast(condition, at::ScalarType::Bool);
  at::Tensor intSelf = op_plugin::npu_dtype_cast(boolSelf, at::ScalarType::Int);
  at::Tensor cout_nonzero_self = at::sum(intSelf, at::ScalarType::Int);
  int64_t nonzero_num = cout_nonzero_self.item().toInt();
  at::SmallVector<int64_t, SIZE> output_size = {nonzero_num, dim};
  return output_size;
}

at::Tensor& where_out_npu_npu_nocheck(
    at::Tensor& out,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor self_cp, other_cp;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_cp = self.to(result_type);
    other_cp = other.to(result_type);
  } else {
    self_cp = self;
    other_cp = other;
  }
  if (condition.scalar_type() != at::ScalarType::Byte && condition.scalar_type() != at::ScalarType::Bool) {
    TORCH_CHECK(false, "Expected condition to have ScalarType Byte, but got ScalarType ",
                toString(condition.scalar_type()));
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("Select")
      .Input(condition)
      .Input(self_cp)
      .Input(other_cp)
      .Output(out)
      .Run();

  return out;
}
} // namespace

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
      where_out_npu_npu_nocheck(contiguous_out, condition, self, other);
    npu_utils::format_fresh_view(out, contiguous_out);
  } else {
    where_out_npu_npu_nocheck(out, condition, self, other);
  }

  return out;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
  at::Tensor ret = npu_preparation::ApplyTensor(b_self);
  where_out_npu_npu_nocheck(ret, b_condition, b_self, b_other);
  return ret;
}

std::vector<at::Tensor> where(const at::Tensor& condition) {
  at::Tensor format_cast_of_condition = condition;
  if (calcu_op_util::GetTensorNpuFormat(condition) != ACL_FORMAT_ND) {
    format_cast_of_condition = at_npu::native::NPUNativeFunctions::npu_format_cast(format_cast_of_condition, ACL_FORMAT_ND);
  }
  if (condition.scalar_type() == at::ScalarType::Half) {
    format_cast_of_condition = op_plugin::npu_dtype_cast(format_cast_of_condition, at::ScalarType::Float);
  }

  auto output_size = where_npu_output_size(format_cast_of_condition);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size, format_cast_of_condition.options().dtype(at::kLong), ACL_FORMAT_ND);

  at_npu::native::OpCommand cmd;
  cmd.Name("NonZero")
      .Input(format_cast_of_condition)
      .Output(result)
      .Run();
  result = result.transpose(1, 0);
  std::vector<at::Tensor> chunk_result = result.chunk(result.size(0), 0);
  std::vector<at::Tensor> squeeze_result;
  for(int64_t i = 0; i < chunk_result.size(); i++){
    squeeze_result.push_back(chunk_result[i].squeeze(0));
  }

  return squeeze_result;
}
} // namespace op_plugin
