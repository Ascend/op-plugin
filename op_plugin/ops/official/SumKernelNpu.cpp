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

namespace{
at::Tensor& sum_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  at::dim_list_to_bitset(dim, self.dim());
  c10::SmallVector<int64_t, N> dim_list = dim.empty() ? calcu_op_util::GetDimlistForTensor(self) :
      c10::SmallVector<int64_t, N>(dim);
  at_npu::native::OpCommand cmd;
  cmd.Name("ReduceSum")
      .Input(self)
      .Input(dim_list, at::kLong)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}
} // namespace

at::Tensor& sum_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& result) {
  auto output_size = op_infer::sum_npu_output_size(self, dim.value(), keepdim);
  auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();

  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      res_type,
      output_size);

  auto self_size = self.sizes();
  for (int64_t i = 0; i < self_size.size(); i++) {
    if (self_size[i] == 0) {
      at::Tensor result_cast = at::empty(output_size);
      result.copy_(result_cast);
      return result;
    }
  }

  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      op_plugin::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor result_cp = result.scalar_type() == self_cp.scalar_type() ? result :
      op_plugin::npu_dtype_cast(result, self_cp.scalar_type());

  if (!npu_utils::check_match(&result_cp)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
    sum_out_npu_nocheck(contiguous_result, self_cp, dim.value(), keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    sum_out_npu_nocheck(result_cp, self_cp, dim.value(), keepdim);
  }

  if (result_cp.scalar_type() != res_type) {
    result_cp = op_plugin::npu_dtype_cast(result_cp, res_type);
    result.copy_(result_cp);
  } else {
    result = result_cp;
  }
  return result;
}

at::Tensor& sum_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& result) {
  return op_plugin::sum_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor sum(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      op_plugin::npu_dtype_cast(self, at::kFloat) : self;
  auto output_size = op_infer::reduce_ops_npu_output_size(self_cp, dim.value(), keepdim);
  auto self_size = self_cp.sizes();
  auto out_type = self.scalar_type();

  if (dtype.has_value()) {
    out_type = dtype.value();
  } else if (isIntegralType(out_type, true)) {
    out_type = at::kLong;
  }

  for (int64_t i = 0; i < self_size.size(); i++) {
    if (self_size[i] == 0) {
      return at::zeros(output_size, self_cp.options());
    }
  }

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size, self_cp.options(), ACL_FORMAT_ND);
  sum_out_npu_nocheck(result, self_cp, dim.value(), keepdim);

  if (result.scalar_type() != out_type) {
    result = op_plugin::npu_dtype_cast(result, out_type);
  }
  return result;
}

at::Tensor sum(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  return op_plugin::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor sum(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
  return op_plugin::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}
} // namespace op_plugin
