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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& lt_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == at::ScalarType::Int || other.dtype() == at::ScalarType::Int ||
      self.dtype() == at::ScalarType::Bool || other.dtype() == at::ScalarType::Bool ||
      self.dtype() == at::ScalarType::Long || other.dtype() == at::ScalarType::Long) {
    self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
    other_cast = at_npu::native::custom_ops::npu_dtype_cast(other, at::ScalarType::Float);
  }
  auto unified_result = npu_preparation::comparison_op_check(result, self_cast, other_cast, true);

  at_npu::native::OpCommand cmd;
  cmd.Name("Less")
      .Expect(unified_result)
      .Input(self_cast)
      .Input(other_cast)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& lt_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  at::Tensor self_cast = self;
  if (self.dtype() == at::ScalarType::Int ||
      self.dtype() == at::ScalarType::Long ||
      self.dtype() == at::ScalarType::Bool) {
    self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("Less")
      .Input(self_cast)
      .Input(other, self_cast.scalar_type())
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor& lt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  at::Tensor format_cast_of_other = npu_preparation::cast_to_ori_format(other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

  npu_preparation::CheckOut(
      {self, other},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    lt_out_npu_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    lt_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
  }

  return result;
}

at::Tensor& lt_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  auto output_size = format_cast_of_self.sizes();
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    lt_out_npu_nocheck(contiguous_result, format_cast_of_self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    lt_out_npu_nocheck(result, format_cast_of_self, other);
  }

  return result;
}

at::Tensor lt(const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return acl_op::lt(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return acl_op::gt(other, self.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
    at::Tensor format_cast_of_other = npu_preparation::cast_to_ori_format(other);

    auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_with_sizes(
        output_size,
        format_cast_of_self.options().dtype(at::kBool));

    // calculate the output result of the NPU
    lt_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor lt(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  auto output_size = op_infer::input_same_output_size(format_cast_of_self);

  at::Tensor result = npu_preparation::apply_tensor_with_sizes(
      output_size,
      format_cast_of_self.options().dtype(at::kBool));

  lt_out_npu_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor& lt_(at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return acl_op::lt_(self, other.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    npu_preparation::cast_to_ori_format(self);
    npu_preparation::cast_to_ori_format(other);
    npu_preparation::CheckMemory({self, other}, {self});

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        self.sizes(),
        self.options().dtype(at::ScalarType::Byte),
        calcu_op_util::GetTensorNpuFormat(self));

    if (!npu_utils::check_match(&self)) {
      at::Tensor contiguous_self = npu_utils::format_contiguous(self);
      lt_out_npu_nocheck(result, contiguous_self, other);
    } else {
      lt_out_npu_nocheck(result, self, other);
    }

    // uint8 to self dtype
    self.copy_(result);
    return self;
  }
}

at::Tensor& lt_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::cast_to_ori_format(self);
  npu_preparation::CheckMemory({self}, {self});

  at::Tensor result = npu_preparation::apply_tensor_with_format(
      self.sizes(),
      self.options().dtype(at::ScalarType::Byte),
      calcu_op_util::GetTensorNpuFormat(self));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    lt_out_npu_nocheck(result, contiguous_self, other);
  } else {
    lt_out_npu_nocheck(result, self, other);
  }

  self.copy_(result);
  return self;
}
} // namespace acl_op
