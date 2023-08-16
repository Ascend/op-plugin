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
std::tuple<at::Tensor&, at::Tensor&> std_mean_out_nocheck(
    at::Tensor& result_std,
    at::Tensor& result_mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  at_npu::native::OpCommand cmd1;
  cmd1.Name("ReduceMeanD")
      .Input(self)
      .Output(result_mean)
      .Attr("axes", dim)
      .Attr("keep_dims", keepdim)
      .Run();

  at::Tensor result_mean_copy = result_mean;
  if (result_mean.dim() != 0 && keepdim == false) {
    auto dim_vector = op_infer::array_to_small_vector(dim);
    std::sort(dim_vector.begin(), dim_vector.end());
    for (int64_t i = 0; i < dim_vector.size(); i++) {
      result_mean_copy = result_mean_copy.unsqueeze(dim_vector[i]);
    }
  }
  result_mean_copy = result_mean_copy.expand(self.sizes());
  at_npu::native::OpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(self)
      .Input(result_mean_copy)
      .Output(result_std)
      .Attr("dim", dim)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Run();

  return std::tie(result_std, result_mean);
}
} // namespace

at::Tensor& std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim.value(), keepdim);
  at::Tensor mean_result = npu_preparation::ApplyTensor(self, output_size);

  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    std_mean_out_nocheck(contiguous_result, mean_result, self, dim.value(), unbiased, keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    std_mean_out_nocheck(result, mean_result, self, dim.value(), unbiased, keepdim);
  }

  return result;
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return acl_op::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

at::Tensor std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim.value(), keepdim);

  at::Tensor result1 = npu_preparation::ApplyTensor(self, output_size);
  at::Tensor result2 = npu_preparation::ApplyTensor(self, output_size);

  std_mean_out_nocheck(result1, result2, self, dim.value(), unbiased, keepdim);
  return result1;
}

at::Tensor std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim) {
  const auto correction_double = correction.value_or(1).toDouble();
  return acl_op::std(self, dim, correction_double > 0, keepdim);
}

at::Tensor std(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return acl_op::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

at::Tensor std(
    const at::Tensor& self,
    bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
  return acl_op::std(self, dims, unbiased, false);
}

std::tuple <at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim.value(), keepdim);

  at::Tensor result1 = npu_preparation::ApplyTensor(self, output_size);
  at::Tensor result2 = npu_preparation::ApplyTensor(self, output_size);

  std_mean_out_nocheck(result1, result2, self, dim.value(), unbiased, keepdim);

  return std::tie(result1, result2);
}

std::tuple <at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
  return acl_op::std_mean(self, dims, unbiased, false);
}

std::tuple <at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return acl_op::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}
} // namespace acl_op
