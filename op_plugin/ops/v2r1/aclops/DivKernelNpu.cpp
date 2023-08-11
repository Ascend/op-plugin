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
at::Tensor& div_scalar_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("RealDiv")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& div_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    div_scalar_out_nocheck(result, self, other.item());
  } else {
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("RealDiv")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

void div_torch_check(c10::optional<c10::string_view> rounding_mode) {
  TORCH_CHECK(false,
      "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
      "but found '", *rounding_mode, "'");
}
} // namespace

at::Tensor& div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor output_tensor = calcu_op_util::IsScalarWrappedToTensor(self) ? other : self;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  if (isFloatingType(result.scalar_type())) {
    high_type = result.scalar_type();
  }
  npu_preparation::CheckOut(
      {self},
      result,
      calcu_op_util::GetTensorNpuFormat(output_tensor),
      high_type,
      output_size);
  at::Tensor self_copy = (self.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(self) &&
      torch_npu::utils::is_npu(self)) ? op_plugin::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(other) &&
      torch_npu::utils::is_npu(other)) ? op_plugin::npu_dtype_cast(other, high_type) : other;
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    div_out_nocheck(contiguous_result, self_copy, other_copy);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    div_out_nocheck(result, self_copy, other_copy);
  }
  return result;
}

at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    at::Tensor& result) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    op_plugin::floor_divide_out(self, other, result);
    return result;
  }
  op_plugin::div_out(self, other, result);
  if (!rounding_mode.has_value()) {
    return result;
  } else if (*rounding_mode == "trunc") {
    op_plugin::trunc_(result);
    return result;
  }
  div_torch_check(rounding_mode);
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::ScalarType::Float;
  }
  at::Tensor self_copy = (self.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(self) &&
      torch_npu::utils::is_npu(self)) ? op_plugin::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !calcu_op_util::IsScalarWrappedToTensor(other) &&
      torch_npu::utils::is_npu(other)) ? op_plugin::npu_dtype_cast(other, high_type) : other;

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options().dtype(high_type),
      calcu_op_util::GetTensorNpuFormat(output_tensor));
  div_out_nocheck(result, self_copy, other_copy);

  return result;
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  div_scalar_out_nocheck(result, self, other);

  return result;
}

at::Tensor div(
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return op_plugin::floor_divide(self, other);
  }
  at::Tensor true_div_res = op_plugin::div(self, other);
  if (!rounding_mode.has_value()) {
    return true_div_res;
  } else if (*rounding_mode == "trunc") {
    return op_plugin::trunc(true_div_res);
  }
  div_torch_check(rounding_mode);
}

at::Tensor div(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return op_plugin::floor_divide(self, other);
  }
  at::Tensor true_div_res = op_plugin::div(self, other);
  if (!rounding_mode.has_value()) {
    return true_div_res;
  } else if (*rounding_mode == "trunc") {
    return op_plugin::trunc(true_div_res);
  }

  div_torch_check(rounding_mode);
}

at::Tensor& div_(at::Tensor& self, const at::Tensor& other) {
  npu_preparation::CheckMemory({self, other}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    op_plugin::div_out(contiguous_self, other, contiguous_self);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    div_out_nocheck(self, self, other);
  }
  return self;
}

at::Tensor& div_(at::Tensor& self, const at::Scalar& other) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    div_scalar_out_nocheck(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    div_scalar_out_nocheck(self, self, other);
  }
  return self;
}

at::Tensor& div_(
    at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return op_plugin::floor_divide_(self, other);
  }
  op_plugin::div_(self, other);
  if (!rounding_mode.has_value()) {
    return self;
  } else if (*rounding_mode == "trunc") {
    return op_plugin::trunc_(self);
  }
  div_torch_check(rounding_mode);
}

at::Tensor& div_(
    at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return op_plugin::floor_divide_(self, other);
  }
  op_plugin::div_(self, other);
  if (!rounding_mode.has_value()) {
    return self;
  } else if (*rounding_mode == "trunc") {
    return op_plugin::trunc_(self);
  }
  div_torch_check(rounding_mode);
}
} // namespace op_plugin
