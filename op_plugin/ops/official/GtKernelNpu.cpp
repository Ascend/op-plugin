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
at::Tensor& gt_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);

  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == at::ScalarType::Bool || other.dtype() == at::ScalarType::Bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
    other_cast = op_plugin::npu_dtype_cast(other, at::ScalarType::Float);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("Greater")
      .Expect(unified_result)
      .Input(self_cast)
      .Input(other_cast)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& gt_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  at::Tensor self_cast = self;
  if (self.dtype() == at::ScalarType::Bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("Greater")
      .Input(self_cast)
      .Input(other, self_cast.scalar_type())
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor& gt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  at::Tensor format_cast_of_other = npu_preparation::CastBackToOriFormat(other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    gt_out_npu_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    gt_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
  }

  return result;
}

at::Tensor& gt_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  auto output_size = format_cast_of_self.sizes();
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    gt_out_npu_nocheck(contiguous_result, format_cast_of_self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    gt_out_npu_nocheck(result, format_cast_of_self, other);
  }

  return result;
}

at::Tensor gt(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  at::Tensor format_cast_of_other = npu_preparation::CastBackToOriFormat(other);

  auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      format_cast_of_self.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  gt_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
  return result;
}

at::Tensor gt(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  auto output_size = op_infer::input_same_output_size(format_cast_of_self);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      format_cast_of_self.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  gt_out_npu_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor& gt_(at::Tensor& self, const at::Tensor& other) {
  npu_preparation::CastBackToOriFormat(self);
  at::Tensor ori_other = npu_preparation::CastBackToOriFormat(other);
  c10::SmallVector<at::Tensor, N> inputs = {self, ori_other};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  calcu_op_util::CheckMemoryOverLaps(inputs, outputs);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      self.sizes(),
      self.options().dtype(at::ScalarType::Byte),
      calcu_op_util::GetTensorNpuFormat(self));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    gt_out_npu_nocheck(result, contiguous_self, ori_other);
  } else {
    gt_out_npu_nocheck(result, self, ori_other);
  }

  self.copy_(result);
  return self;
}

at::Tensor& gt_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::CastBackToOriFormat(self);
  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  calcu_op_util::CheckMemoryOverLaps(inputs, outputs);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      self.sizes(),
      self.options().dtype(at::ScalarType::Byte),
      calcu_op_util::GetTensorNpuFormat(self));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    gt_out_npu_nocheck(result, contiguous_self, other);
  } else {
    gt_out_npu_nocheck(result, self, other);
  }

  self.copy_(result);
  return self;
}
} // namespace op_plugin
