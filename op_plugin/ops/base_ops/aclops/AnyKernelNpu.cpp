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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& any_out_npu_nocheck(
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
    dim_list = op_plugin::utils::get_dimlist_for_tensor(self);
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

at::Tensor& any_out(
    const at::Tensor& self,
    at::Tensor& result) {
  // when self's dim = 0, convert [1] tensor and reduce it
  if (self.dim() == 0) {
    at::Tensor self_tmp = self.unsqueeze(0);
    self_tmp = at_npu::native::custom_ops::npu_dtype_cast(self_tmp, at::kBool);
    return acl_op::any_out(self_tmp, 0, false, result);
  }

  at::SmallVector<int64_t, N> dim_list = op_plugin::utils::get_dimlist_for_tensor(self);
  bool keepdim = false;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (self.numel() == 0) {
    result.fill_(false);
    return result;
  }

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
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      output_size, self.options(), npu_preparation::get_tensor_npu_format(self));

  if (dim == LLONG_MIN) {
    any_out_npu_nocheck(
        result, self, op_plugin::utils::get_dimlist_for_tensor(self), keepdim);
  } else {
    any_out_npu_nocheck(result, self, {dim}, keepdim);
  }
  return result;
}

at::Tensor any(const at::Tensor& self) {
  // when self's dim = 0, convert [1] tensor and reduce it.
  if (self.dim() == 0) {
    at::Tensor self_tmp = self.unsqueeze(0);
    self_tmp = at_npu::native::custom_ops::npu_dtype_cast(self_tmp, at::kBool);
    return acl_op::any(self_tmp, 0, false);
  }

  at::IntArrayRef dims;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  any_out_npu_nocheck(
      result, self, op_plugin::utils::get_dimlist_for_tensor(self), false);
  return result;
}
} // namespace acl_op
