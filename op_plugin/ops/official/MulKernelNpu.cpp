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
at::Tensor& muls_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  if (!other.isFloatingPoint()) {
    unified_result.common_type = self.scalar_type();
    if (self.scalar_type() == at::kBool) {
      unified_result.common_type = other.type();
    }
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("Mul")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& mul_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    muls_out_npu_nocheck(result, self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    muls_out_npu_nocheck(result, other, self.item());
  } else {
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("Mul")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor mul_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  return is_self_wrapped ? other : self;
}
} // namespace

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor output_tensor = mul_dest_output(self, other);
  auto high_type = output_tensor.scalar_type();
  auto result_type = result.scalar_type();
  TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
      " can't be cast to the desired output type ", result_type);

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      result,
      output_size);
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == at::kBool && other.dtype() == at::kBool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::kFloat);
    other_cast = op_plugin::npu_dtype_cast(other, at::kFloat);
  }

  bool result_is_cast = (result.scalar_type() != self.scalar_type());
  at::Tensor result_cast = result_is_cast ? op_plugin::npu_dtype_cast(result, self.scalar_type()) : result;

  if (!npu_utils::check_match(&result_cast)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
    mul_out_npu_nocheck(contiguous_result, self_cast, other_cast);
    npu_utils::format_fresh_view(result_cast, contiguous_result);
  } else {
    mul_out_npu_nocheck(result_cast, self_cast, other_cast);
  }

  if (result_is_cast) {
    result_cast = op_plugin::npu_dtype_cast(result_cast, result.scalar_type());
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  bool self_other_is_bool =
      (self.dtype() == c10::ScalarType::Bool && other.dtype() == c10::ScalarType::Bool);
  if (self_other_is_bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::kFloat);
    other_cast = op_plugin::npu_dtype_cast(other, at::kFloat);
  }

  at::Tensor output_tensor = mul_dest_output(self_cast, other_cast);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_cast, other_cast);
  at::Tensor result = npu_preparation::ApplyTensor(output_tensor, output_size);

  mul_out_npu_nocheck(result, self_cast, other_cast);

  if (self_other_is_bool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }

  return result;
}

at::Tensor mul(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  muls_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK(torch_npu::utils::is_npu(self), "input self must be NPU-Tensor");
  npu_preparation::CheckMemory({self, other}, {self});

  at::Tensor self_dtype_cast =
      (self.scalar_type() == at::kBool) ? op_plugin::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor other_dtype_cast =
      (other.scalar_type() == at::kBool) ? op_plugin::npu_dtype_cast(other, at::kFloat) : other;

  if (!npu_utils::check_match(&self_dtype_cast)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self_dtype_cast);
    mul_out_npu_nocheck(contiguous_self, contiguous_self, other_dtype_cast);
    npu_utils::format_fresh_view(self_dtype_cast, contiguous_self);
  } else {
    mul_out_npu_nocheck(self_dtype_cast, self_dtype_cast, other_dtype_cast);
  }

  if (self_dtype_cast.scalar_type() != self.scalar_type()) {
    self_dtype_cast = op_plugin::npu_dtype_cast(self_dtype_cast, self.scalar_type());
    self.copy_(self_dtype_cast);
  } else {
    self = self_dtype_cast;
  }
  return self;
}

at::Tensor& mul_(at::Tensor& self, const at::Scalar& other) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    muls_out_npu_nocheck(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    muls_out_npu_nocheck(self, self, other);
  }
  return self;
}
} // namespace op_plugin
