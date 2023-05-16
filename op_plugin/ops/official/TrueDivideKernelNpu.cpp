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
at::Tensor& true_div_scalar_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("Div")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& true_div_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  // executing the NPU operator
  if (other.dim() == 0) {
    true_div_scalar_out_npu(result, self, other.item());
  } else {
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("Div")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}
} // namespace

at::Tensor& true_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor self_temp = self;
  at::Tensor other_temp = other;
  auto high_dtype = at::native::result_type(self, other);
  auto result_dtype = result.scalar_type();
  TORCH_CHECK(canCast(high_dtype, result_dtype),
      "result type ", high_dtype, " can't be cast to the desired output type ", result_dtype);
  if (isIntegralType(high_dtype, true)) {
      high_dtype = at::ScalarType::Float;
  }
  if (isFloatingType(result_dtype)) {
      high_dtype = result_dtype;
  }
  at::Tensor output_tensor = calcu_op_util::IsScalarWrappedToTensor(self_temp) ? other_temp : self_temp;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_temp, other_temp);
  npu_preparation::CheckOut(
      {self_temp},
      result,
      calcu_op_util::GetTensorNpuFormat(output_tensor),
      high_dtype,
      output_size);

  if (self.scalar_type() != result.scalar_type()) {
    self_temp = op_plugin::npu_dtype_cast(self, result.scalar_type());
    other_temp = other.to(result.scalar_type());
  }

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(self_temp);
    true_div_out_npu_nocheck(contiguous_result, self_temp, other_temp);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    true_div_out_npu_nocheck(result, self_temp, other_temp);
  }

  return result;
}

at::Tensor true_divide(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_temp = self;
  at::Tensor other_temp = other;
  if (self.scalar_type() == at::ScalarType::Int || self.scalar_type() == at::ScalarType::Bool) {
    self_temp = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  if (other.scalar_type() == at::ScalarType::Int) {
    other_temp = other.to(at::ScalarType::Float);
  }

  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self_temp);
  at::Tensor output_tensor = is_self_wrapped ? other_temp : self_temp;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_temp, other_temp);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options(),
      calcu_op_util::GetTensorNpuFormat(output_tensor));

  true_div_out_npu_nocheck(result, self_temp, other_temp);

  return result;
}

at::Tensor true_divide(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  true_div_scalar_out_npu(result, self, other);

  return result;
}

at::Tensor& true_divide_(at::Tensor& self, const at::Tensor& other) {
  at::Tensor other_temp = other;
  if (self.scalar_type() != other.scalar_type()) {
    other_temp = other.to(self.scalar_type());
  }
  op_plugin::true_divide_out(self, other_temp, self);

  return self;
}

at::Tensor& true_divide_(at::Tensor& self, const at::Scalar& other) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    true_div_scalar_out_npu(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    true_div_scalar_out_npu(self, self, other);
  }
  return self;
}
} // namespace op_plugin
