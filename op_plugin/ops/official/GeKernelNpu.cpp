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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& ge_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_cast = self;
  at::Tensor other_cast = other;
  if (self.dtype() == at::ScalarType::Int || other.dtype() == at::ScalarType::Int ||
      self.dtype() == at::ScalarType::Bool || other.dtype() == at::ScalarType::Bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
    other_cast = op_plugin::npu_dtype_cast(other, at::ScalarType::Float);
  }
  auto unified_result = npu_preparation::comparison_op_check(result, self_cast, other_cast, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("GreaterEqual")
      .Expect(unified_result)
      .Input(self_cast)
      .Input(other_cast)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& ge_out_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  at::Tensor self_cast = self;
  if (self.dtype() == at::ScalarType::Int || self.dtype() == at::ScalarType::Bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("GreaterEqual")
     .Input(self_cast)
     .Input(other, self_cast.scalar_type())
     .Output(result)
     .Run();
  return result;
}
} // namespace

at::Tensor& ge_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  at::Tensor format_cast_of_other = npu_preparation::CastBackToOriFormat(other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

  npu_preparation::CheckOut(
      {self, other},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    ge_out_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    ge_out_nocheck(result, format_cast_of_self, format_cast_of_other);
  }

  return result;
}

at::Tensor& ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
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
    ge_out_nocheck(contiguous_result, format_cast_of_self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    ge_out_nocheck(result, format_cast_of_self, other);
  }
  return result;
}

at::Tensor ge(const at::Tensor& self, const at::Tensor& other) {
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    return op_plugin::ge(self, other.item());
  } else if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
    return op_plugin::lt(other, self.item());
  } else {
    at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
    at::Tensor format_cast_of_other = npu_preparation::CastBackToOriFormat(other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, format_cast_of_self.options().dtype(at::kBool), ACL_FORMAT_ND);
    ge_out_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor ge(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      format_cast_of_self.sizes(),
      format_cast_of_self.options().dtype(at::kBool),
      ACL_FORMAT_ND);
  ge_out_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor& ge_(at::Tensor& self, const at::Tensor& other) {
  npu_preparation::CastBackToOriFormat(self);
  at::Tensor ori_other = npu_preparation::CastBackToOriFormat(other);
  npu_preparation::CheckMemory({self, ori_other}, {self});

  at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(at::ScalarType::Byte));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    ge_out_nocheck(result, contiguous_self, ori_other);
  } else {
    ge_out_nocheck(result, self, ori_other);
  }

  self.copy_(result);
  return self;
}

at::Tensor& ge_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::CastBackToOriFormat(self);
  at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(at::ScalarType::Byte));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    ge_out_nocheck(result, contiguous_self, other);
  } else {
    ge_out_nocheck(result, self, other);
  }

  self.copy_(result);
  return self;
}
} // namespace op_plugin
