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
at::Tensor& floor_divide_out_scalar_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FloorDiv")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& floor_divide_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (other.dim() == 0) {
    floor_divide_out_scalar_nocheck(result, self, other.item());
  } else {
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorDiv")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& check_self_dtype_npu(at::Tensor& self){
  if (self.dtype() == at::kBool || self.dtype() == at::kInt) {
    self = op_plugin::npu_dtype_cast(self, at::kFloat);
  }
  return self;
}

std::tuple<at::Tensor, at::Tensor> check_dtype_npu(at::Tensor& self, at::Tensor& other){
  if (self.dtype() == at::kBool || (self.dtype() == at::kInt && other.scalar_type() == at::kDouble)) {
    self = op_plugin::npu_dtype_cast(self, at::kFloat);
  }
  if (other.scalar_type() == at::kDouble) {
    other = other.to(at::kFloat);
  }
  if (other.scalar_type() == at::kLong) {
    other = other.to(at::kInt);
  }
  return std::tie(self, other);
}
} // namespace

at::Tensor& floor_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  check_dtype_npu(self_cast, other_cast);
  at::Tensor format_cast_self = npu_preparation::CastBackToOriFormat(self_cast);
  auto output_size = format_cast_self.sizes();
  npu_preparation::CheckOut(
      {self_cast, other_cast}, result,
      calcu_op_util::GetTensorNpuFormat(self_cast),
      result.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    floor_divide_out_nocheck(contiguous_result, format_cast_self, other_cast);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    floor_divide_out_nocheck(result, format_cast_self, other_cast);
  }
  return result;
}

at::Tensor floor_divide(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  check_dtype_npu(self_cast, other_cast);
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self_cast);
  at::Tensor output_tensor = is_self_wrapped ? other_cast : self_cast;
  at::Tensor format_cast_self = npu_preparation::CastBackToOriFormat(self_cast);
  auto output_size = format_cast_self.sizes();
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options(),
      calcu_op_util::GetTensorNpuFormat(self_cast));
  floor_divide_out_nocheck(result, format_cast_self, other_cast);
  return result;
}

at::Tensor floor_divide(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor self_cast = self;
  check_self_dtype_npu(self_cast);
  at::Tensor format_cast_self = npu_preparation::CastBackToOriFormat(self_cast);
  auto output_size = format_cast_self.sizes();
  at::Tensor result = npu_preparation::ApplyTensor(self_cast, output_size);
  floor_divide_out_scalar_nocheck(result, format_cast_self, other);
  return result;
}

at::Tensor& floor_divide_(at::Tensor& self, const at::Tensor& other) {
  at::Tensor other_cast = other;
  check_dtype_npu(self, other_cast);
  npu_preparation::CheckMemory({self, other_cast}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    floor_divide_out_nocheck(contiguous_self, contiguous_self, other_cast);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    floor_divide_out_nocheck(self, self, other_cast);
  }
  return self;
}

at::Tensor& floor_divide_(at::Tensor& self, const at::Scalar& other) {
  check_self_dtype_npu(self);
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    floor_divide_out_scalar_nocheck(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    floor_divide_out_scalar_nocheck(self, self, other);
  }
  return self;
}
} // namespace op_plugin
