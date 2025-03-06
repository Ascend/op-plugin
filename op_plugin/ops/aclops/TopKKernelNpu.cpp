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
std::tuple<at::Tensor&, at::Tensor&> topk_out_npu_no_transpose(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  c10::SmallVector<int64_t, N> k_vec = {k};
  at_npu::native::OpCommand cmd;
  cmd.Name("TopKV2")
      .Input(self)
      .Input(k_vec, at::kInt)
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Attr("largest", largest)
      .Attr("sorted", sorted)
      .Run();
  return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> topk_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  dim = op_plugin::utils::make_warp_dim(dim, self.dim());
  int64_t last_dim = op_plugin::utils::make_warp_dim(-1, self.dim());
  if (dim != last_dim) {
    c10::SmallVector<int64_t, SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[last_dim]);

    at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);
    auto output_size = op_infer::transpose_npu_output_size(values, perm);
    at::Tensor transpose_value = npu_preparation::apply_tensor(values, output_size);
    at::Tensor transpose_indices = npu_preparation::apply_tensor(indices, output_size);
    topk_out_npu_no_transpose(
        transpose_value,
        transpose_indices,
        transpose_self,
        k,
        last_dim,
        largest,
        sorted);
    acl_op::npu_transpose_out(transpose_value, perm, true, values);
    acl_op::npu_transpose_out(transpose_indices, perm, true, indices);
  } else {
    topk_out_npu_no_transpose(
        values, indices, self, k, last_dim, largest, sorted);
  }

  return std::tie(values, indices);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> topk_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  at::Tensor self_cp = npu_preparation::CastBackToOriFormat(self);
  auto output_size = op_infer::topk_npu_output_size(self_cp, k, dim);
  npu_preparation::CheckOut(
      {self},
      values,
      self,
      output_size);
  npu_preparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      output_size);

  at::Tensor indices_cp = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kInt);
  bool values_match = npu_utils::check_match(&values);
  bool indices_match = npu_utils::check_match(&indices_cp);
  if (!(values_match && indices_match)) {
    at::Tensor contiguous_values = values_match ? values : npu_utils::format_contiguous(values);
    at::Tensor contiguous_indices = indices_match ? indices_cp : npu_utils::format_contiguous(indices_cp);
    topk_out_npu_nocheck(contiguous_values, contiguous_indices, self_cp, k, dim, largest, sorted);
    if (!values_match) {
      npu_utils::format_fresh_view(values, contiguous_values);
    }
    if (!indices_match) {
      npu_utils::format_fresh_view(indices_cp, contiguous_indices);
    }
  } else {
    topk_out_npu_nocheck(values, indices_cp, self_cp, k, dim, largest, sorted);
  }
  // indices dtype transform Int64
  indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kLong);
  indices.copy_(indices_cp);
  return std::tie(values, indices);
}

std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  at::Tensor self_cp = npu_preparation::CastBackToOriFormat(self);
  auto output_size = op_infer::topk_npu_output_size(self_cp, k, dim);
  at::Tensor values = npu_preparation::apply_tensor(self_cp, output_size);
  at::Tensor indices = npu_preparation::apply_tensor_with_format(
      output_size, self_cp.options().dtype(at::kInt), ACL_FORMAT_ND);
  topk_out_npu_nocheck(values, indices, self_cp, k, dim, largest, sorted);

  // indices dtype transform Int64
  indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kLong);

  return std::tie(values, indices);
}
} // namespace acl_op
