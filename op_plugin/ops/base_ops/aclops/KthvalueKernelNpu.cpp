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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> kthvalue_npu_output_size(const at::Tensor& self, int64_t dim, bool keepdim) {
  at::IntArrayRef dims(dim);
  return op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
}

void kthvalue_shape_modify(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::Tensor self_rename = self.rename(c10::nullopt);
  auto output_size = kthvalue_npu_output_size(self, dim, keepdim);
  if (values.defined()) {
    TORCH_CHECK(
        values.dtype() == self.dtype(),
        "output values must be of same type as input"
        + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        values.device() == self.device(),
        "output values must be on same values as input"
        + OPS_ERROR(ErrCode::PARAM));
    values.resize_(output_size);
  } else {
    values = at::empty(output_size, self_rename.options());
  }

  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == at::kLong,
        "output indices must be of scalar type Long"
        + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input"
        + OPS_ERROR(ErrCode::PARAM));
    indices.resize_(output_size);
  } else {
    indices = at::empty(output_size, self_rename.options().dtype(at::kLong));
  }
  return;
}

void kthvalue_calculate(
    const at::Tensor& self,
    at::Tensor& result,
    at::Tensor x,
    int64_t k,
    int64_t dim,
    bool keepdim,
    bool change_type,
    bool is_indices) {
  at::Tensor index = npu_preparation::apply_tensor({1}, self.options().dtype(at::kInt), self);
  acl_op::fill_(index, k - 1);

  at::Tensor y = acl_op::index_select(x, dim, index);
  if (!keepdim) {
    y.squeeze_(dim);
  }

  if (change_type) {
    y = at_npu::native::custom_ops::npu_dtype_cast(y, self.scalar_type());
  }
  if (is_indices) {
    y = at_npu::native::custom_ops::npu_dtype_cast(y, at::kLong);
  }
  result.copy_(y, false);
  at::namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
  return;
}

void check_self_dim(const at::Tensor& self, int64_t k, int64_t dim) {
  TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat || self.scalar_type() == at::kInt,
      "the type of input must be float16, float32, or int32"
      + OPS_ERROR(ErrCode::TYPE));
  dim = op_plugin::utils::make_warp_dim(dim, self.dim());
  TORCH_CHECK(k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1), "selected index k out of range"
      + OPS_ERROR(ErrCode::VALUE));
}

std::tuple<at::Tensor, at::Tensor> kthvalue_out_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  dim = op_plugin::utils::make_warp_dim(dim, self.dim());
  at::Tensor self_rename = self.rename(c10::nullopt);
  kthvalue_shape_modify(values, indices, self, dim, keepdim);

  bool change_type = false;
  if (self.scalar_type() != at::kHalf) {
    change_type = true;
    self_rename = at_npu::native::custom_ops::npu_dtype_cast(self_rename, at::kHalf);
  }
  auto ret = at::topk(self_rename, k, dim, false, true);

  kthvalue_calculate(self, values, std::get<0>(ret), k, dim, keepdim, change_type, false);
  kthvalue_calculate(self, indices, std::get<1>(ret), k, dim, keepdim, false, true);
  return std::tuple<at::Tensor, at::Tensor>(values, indices);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> kthvalue(const at::Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  check_self_dim(self, k, dim);
  auto output_size = kthvalue_npu_output_size(self, dim, keepdim);
  at::Tensor values = npu_preparation::apply_tensor(self, output_size);
  at::Tensor indices =
      npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kLong), ACL_FORMAT_NCHW);
  kthvalue_out_nocheck(values, indices, self, k, dim, keepdim);
  return std::tuple<at::Tensor, at::Tensor>(values, indices);
}

std::tuple<at::Tensor, at::Tensor> kthvalue(const at::Tensor& self, int64_t k, at::Dimname dim, bool keepdim) {
  return acl_op::kthvalue(self, k, dimname_to_position(self, dim), keepdim);
}

std::tuple<at::Tensor&, at::Tensor&> kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  check_self_dim(self, k, dim);
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  npu_preparation::CheckOut(
      {self},
      values,
      npu_preparation::get_tensor_npu_format(values),
      self.scalar_type(),
      output_size);

  npu_preparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      output_size);

  kthvalue_out_nocheck(values, indices, self, k, dim, keepdim);
  return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  return acl_op::kthvalue_out(self, k, dimname_to_position(self, dim), keepdim, values, indices);
}
} // namespace acl_op
