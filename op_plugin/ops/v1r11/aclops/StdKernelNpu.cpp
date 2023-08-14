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
int64_t calc_shape_prod(const at::Tensor& self, at::IntArrayRef dim) {
  int64_t shape_prod = 1;
  if (self.dim() == 0) {
    shape_prod = 1;
  } else if (dim.size() == 0) {
    for (auto i = 0; i < self.dim(); i++) {
      shape_prod *= self.size(i);
    }
  } else {
    for(auto i = 0; i < dim.size(); i++) {
      shape_prod *= self.size(dim[i]);
    }
  }
  return shape_prod;
}

std::tuple<at::Tensor&, at::Tensor&> std_mean_out_npu_nocheck(
    at::Tensor& result_std,
    at::Tensor& result_mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    int64_t correction) {
  at_npu::native::OpCommandOpCommand cmd1;
  cmd1.Name("ReduceMeanD")
      .Input(self)
      .Output(result_mean)
      .Attr("axes", dim)
      .Attr("keep_dims", keepdim)
      .Run();

  auto shape_prod = calc_shape_prod(self, dim);
  if (shape_prod == 0 || (shape_prod == 1 && shape_prod <= correction)) {
    result_std.fill_(NAN);
    return std::tie(result_std, result_mean);
  }
  if (correction > 1 && shape_prod <= correction) {
    result_std.fill_(INFINITY);
    return std::tie(result_std, result_mean);
  }

  at::Tensor result_mean_copy = result_mean;
  if (result_mean.dim() != 0 && keepdim == false) {
    auto dimVector = array_to_small_vector(dim);
    std::sort(dimVector.begin(), dimVector.end());
    for (int64_t i = 0; i < dimVector.size(); i++) {
      result_mean_copy = result_mean_copy.unsqueeze(dimVector[i]);
    }
  }
  result_mean_copy = result_mean_copy.expand(self.sizes());
  at_npu::native::OpCommandOpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(self)
      .Input(result_mean_copy)
      .Output(result_std)
      .Attr("dim", dim)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Attr("correction", correction)
      .Run();

  return std::tie(result_std, result_mean);
}
} // namespace

at::Tensor& std_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return op_plugin::std_out(self, c10::optional<at::IntArrayRef>(dim), int64_t{unbiased ? 1 : 0}, keepdim, result);
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  return op_plugin::std_out(self, dimnames_to_positions(self, dim), correction, keepdim, result);
}

at::Tensor& std_out(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> dims = calcu_op_util::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor mean_result = npu_preparation::apply_tensor(self, output_size);
  auto real_correction = correction.has_value() ? correction.value() : 1;

  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    std_mean_out_npu_nocheck(contiguous_result, mean_result, self, dims, correction.has_value() ? true : false, keepdim, real_correction);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    std_mean_out_npu_nocheck(result, mean_result, self, dims, correction.has_value() ? true : false, keepdim, real_correction);
  }

  return result;
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return op_plugin::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

std::tuple <at::Tensor, at::Tensor> std_mean(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return op_plugin::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple <at::Tensor, at::Tensor> std_mean(
    const at::Tensor & self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = calcu_op_util::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = op_infer::array_to_small_vector(dim.value());
  }

  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

  at::Tensor result1 = npu_preparation::apply_tensor(self, output_size);
  at::Tensor result2 = npu_preparation::apply_tensor(self, output_size);

  auto real_correction = correction.has_value() ? correction.value() : 1;
  std_mean_out_npu_nocheck(result1, result2, self, dims, correction.has_value() ? true : false, keepdim, real_correction);

  return std::tie(result1, result2);
}

at::Tensor std(
    const at::Tensor & self,
    at::DimnameList dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  return op_plugin::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor std(
    const at::Tensor & self,
    c10::optional<at::IntArrayRef> dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = calcu_op_util::GetDimlistForTensor(self);
  if (dim.has_value()) {
    dims = op_infer::array_to_small_vector(dim.value());
  }

  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

  at::Tensor result1 = npu_preparation::apply_tensor(self, output_size);
  at::Tensor result2 = npu_preparation::apply_tensor(self, output_size);

  auto real_correction = correction.has_value() ? correction.value() : 1;
  std_mean_out_npu_nocheck(result1, result2, self, dims, correction.has_value() ? true : false, keepdim, real_correction);
  return result1;
}
} // namespace at_npu
