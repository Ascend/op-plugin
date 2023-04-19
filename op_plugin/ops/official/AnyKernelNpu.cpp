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
inline at::Tensor& any_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dim_list,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ReduceAny")
      .Input(self)
      .Input(dim_list)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
   return result;
}
} // namespace

at::Tensor& any_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> dim_list;
  if (dim == LLONG_MIN) {
    dim_list = calcu_op_util::GetDimlistForTensor(self);
  } else {
    dim_list = {dim};
  }

  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    any_out_npu_nocheck(contiguous_result, self, dim_list, keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    any_out_npu_nocheck(result, self, dim_list, keepdim);
  }
  return result;
}

at::Tensor any(const at::Tensor& self, int64_t dim, bool keepdim) {
  at::IntArrayRef dims(dim);
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size, self.options(), calcu_op_util::GetTensorNpuFormat(self));

  if (dim == LLONG_MIN) {
    any_out_npu_nocheck(
        result, self, calcu_op_util::GetDimlistForTensor(self), keepdim);
  } else {
    any_out_npu_nocheck(result, self, {dim}, keepdim);
  }
  return result;
}

at::Tensor any(const at::Tensor& self) { 
  // when self's dim = 0, convert [1] tensor and reduce it.
  if (self.dim() == 0) {
    at::Tensor self_tmp = npu_preparation::ApplyTensorWithFormat(
        {1}, 
        self.options().dtype(at::ScalarType::Float), 
        calcu_op_util::GetTensorNpuFormat(self));
    op_plugin::fill_(self_tmp, self.item());
    self_tmp = op_plugin::npu_dtype_cast(self_tmp, at::kBool);
    return op_plugin::any(self_tmp, 0, false);
  }

  at::IntArrayRef dims;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  any_out_npu_nocheck(
      result, self, calcu_op_util::GetDimlistForTensor(self), false);
  return result;
}
} // namespace op_plugin