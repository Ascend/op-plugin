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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& div_scalar_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar& other) {
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

at::Tensor& div_scalar_out_nocheck(at::Tensor& result, const at::Scalar& self, const at::Tensor& other) {
  auto unified_result = npu_preparation::binary_op_check(result, other, self, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("RealDiv")
      .Expect(unified_result)
      .Input(self, other.scalar_type())
      .Input(other)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& div_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    div_scalar_out_nocheck(result, self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    div_scalar_out_nocheck(result, self.item(), other);
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
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      result,
      output_size);

  auto result_type = result.scalar_type();
  auto calculate_type = op_plugin::utils::get_divide_high_type(self, other);
  TORCH_CHECK(canCast(calculate_type, result_type),
      "result type ", calculate_type, " can't be cast to the desired output type ", result_type);

  at::Tensor self_cast = (self.scalar_type() == calculate_type) ? self : self.to(calculate_type);
  at::Tensor other_cast = (other.scalar_type() == calculate_type) ? other : other.to(calculate_type);

  at::Tensor result_cast = (result_type == calculate_type) ? result : at_npu::native::custom_ops::npu_dtype_cast(result, calculate_type);
  if (!npu_utils::check_match(&result_cast)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
    div_out_nocheck(contiguous_result, self_cast, other_cast);
    npu_utils::format_fresh_view(result_cast, contiguous_result);
  } else {
    div_out_nocheck(result_cast, self_cast, other_cast);
  }

  if (result_type != calculate_type) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    at::Tensor& result) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    acl_op::floor_divide_out(self, other, result);
    return result;
  }
  acl_op::div_out(self, other, result);
  if (!rounding_mode.has_value()) {
    return result;
  } else if (*rounding_mode == "trunc") {
    acl_op::trunc_(result);
    return result;
  }
  div_torch_check(rounding_mode);
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other) {
  auto high_type = op_plugin::utils::get_divide_high_type(self, other);
  at::Tensor self_temp = (self.scalar_type() == high_type) ? self : self.to(high_type);
  at::Tensor other_temp = (other.scalar_type() == high_type) ? other : other.to(high_type);

  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self_temp) || npu_preparation::IsCPUScalar(self_temp);
  at::Tensor output_tensor = is_self_wrapped ? other_temp : self_temp;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_temp, other_temp);

  at::Tensor result = npu_preparation::apply_tensor(output_tensor, output_size);
  div_out_nocheck(result, self_temp, other_temp);

  return result;
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  div_scalar_out_nocheck(result, self, other);

  return result;
}

at::Tensor div(
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return acl_op::floor_divide(self, other);
  }
  at::Tensor true_div_res = acl_op::div(self, other);
  if (!rounding_mode.has_value()) {
    return true_div_res;
  } else if (*rounding_mode == "trunc") {
    return acl_op::trunc(true_div_res);
  }
  div_torch_check(rounding_mode);
}

at::Tensor div(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    return acl_op::floor_divide(self, other);
  }
  at::Tensor true_div_res = acl_op::div(self, other);
  if (!rounding_mode.has_value()) {
    return true_div_res;
  } else if (*rounding_mode == "trunc") {
    return acl_op::trunc(true_div_res);
  }

  div_torch_check(rounding_mode);
}

at::Tensor& div_(at::Tensor& self, const at::Tensor& other) {
  return acl_op::div_out(self, other, self);
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
    return acl_op::floor_divide_(self, other);
  }
  acl_op::div_(self, other);
  if (!rounding_mode.has_value()) {
    return self;
  } else if (*rounding_mode == "trunc") {
    return acl_op::trunc_(self);
  }
  div_torch_check(rounding_mode);
}

at::Tensor& div_(
    at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  return acl_op::div_out(self, other, rounding_mode, self);
}
} // namespace acl_op
