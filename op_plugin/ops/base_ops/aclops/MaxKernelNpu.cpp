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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&> max_out_npu_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}

at::Tensor& max_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Maximum")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& max_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ReduceMax")
      .Input(self)
      .Input(dims)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}

at::Tensor& max_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Maximum")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> max_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {

  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

  npu_preparation::CheckOut(
      {self},
      output,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  npu_preparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      output_size);

  at::Tensor indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(indices, at::ScalarType::Int);
  bool output_match = npu_utils::check_match(&output);
  bool indices_match = npu_utils::check_match(&indices_dtype_cast);

  if (!(output_match && indices_match)) {
    at::Tensor contiguous_output = output_match ? output : npu_utils::format_contiguous(output);
    at::Tensor contiguous_indices =
        indices_match ? indices_dtype_cast : npu_utils::format_contiguous(indices_dtype_cast);

    max_out_npu_nocheck(contiguous_output, contiguous_indices, self, dim, keepdim);

    if (!output_match) {
      npu_utils::format_fresh_view(output, contiguous_output);
    }
    if (!indices_match) {
      npu_utils::format_fresh_view(indices_dtype_cast, contiguous_indices);
    }
  } else {
    max_out_npu_nocheck(output, indices_dtype_cast, self, dim, keepdim);
  }

  indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(indices_dtype_cast, at::ScalarType::Long);
  indices.copy_(indices_dtype_cast);
  return std::tie(output, indices);
}

std::tuple<at::Tensor&, at::Tensor&> max_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  return acl_op::max_out(self, dimname_to_position(self, dim), keepdim, output, indices);
}

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, int64_t dim, bool keepdim) {
  at::Tensor self_cast = self;
  if (self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int) {
    self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self_cast, dims, keepdim);

  at::Tensor outputs = npu_preparation::ApplyTensorWithFormat(output_size, self_cast.options(), ACL_FORMAT_ND);
  at::Tensor indices = npu_preparation::ApplyTensorWithFormat(
      output_size,
      self_cast.options().dtype(at::ScalarType::Int),
      ACL_FORMAT_ND);

  max_out_npu_nocheck(outputs, indices, self_cast, dim, keepdim);
  indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::ScalarType::Long);

  if (self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int) {
    outputs = at_npu::native::custom_ops::npu_dtype_cast(outputs, self.scalar_type());
  }

  return std::tie(outputs, indices);
}

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor& max_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_copy = (self.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(self)) ?
      at_npu::native::custom_ops::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(other)) ?
      at_npu::native::custom_ops::npu_dtype_cast(other, high_type) : other;

  npu_preparation::CheckOut(
      {self_copy, other_copy},
      result,
      self_copy,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    max_out_npu_nocheck(contiguous_result, self_copy, other_copy);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    max_out_npu_nocheck(result, self_copy, other_copy);
  }

  return result;
}

at::Tensor& maximum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    max_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    max_out_npu_nocheck(result, self, other);
  }

  return result;
}

at::Tensor maximum(const at::Tensor& self, const at::Tensor& other) {
  auto output_size_diff = self.sizes();
  at::Tensor result_diff = npu_preparation::ApplyTensor(self, output_size_diff);
  if (npu_preparation::IsCPUScalar(other)) {
    max_out_npu_nocheck(result_diff, self, other.item());
    return result_diff;
  }
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_copy = (self.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(self)) ?
      at_npu::native::custom_ops::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(other)) ?
      at_npu::native::custom_ops::npu_dtype_cast(other, high_type) : other;
  at::Tensor result = npu_preparation::ApplyTensor(self_copy, output_size);
  max_out_npu_nocheck(result, self_copy, other_copy);
  return result;
}

at::Tensor amax(const at::Tensor& self, at::IntArrayRef dims, bool keepdim) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = calcu_op_util::GetTensorNpuFormat(self);
  if (output_size.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(self, output_size, npu_format);
  max_out_npu_nocheck(result, self, dims, keepdim);
  return result;
}

at::Tensor max(const at::Tensor& self) {
  at::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
  return acl_op::amax(self, dims, false);
}

at::Tensor& amax_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    max_out_npu_nocheck(contiguous_result, self, dims, keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    max_out_npu_nocheck(result, self, dims, keepdim);
  }

  return result;
}
} // namespace acl_op
