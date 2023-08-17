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
at::Tensor& mul_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar& other) {
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
    mul_out_npu_nocheck(result, self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    mul_out_npu_nocheck(result, other, self.item());
  } else {
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("Mul")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
    return result;
  }
}
} // namespace

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self) || npu_preparation::IsCPUScalar(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;
  auto result_type = result.scalar_type();
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      calcu_op_util::GetTensorNpuFormat(output_tensor),
      result_type,
      output_size);

  auto calculate_type = at::native::result_type(self, other);
  TORCH_CHECK(canCast(calculate_type, result_type),
      "result type ", calculate_type, " can't be cast to the desired output type ", result_type);

  if (calculate_type == at::kBool) {
    calculate_type = at::kFloat;
  }
  at::Tensor self_cast = (self.scalar_type() == calculate_type) ? self : self.to(calculate_type);
  at::Tensor other_cast = (other.scalar_type() == calculate_type) ? other : other.to(calculate_type);

  at::Tensor result_cast = (result_type == calculate_type) ? result :
      at_npu::native::custom_ops::npu_dtype_cast(result, calculate_type);
  if (!npu_utils::check_match(&result_cast)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
    mul_out_npu_nocheck(contiguous_result, self_cast, other_cast);
    npu_utils::format_fresh_view(result_cast, contiguous_result);
  } else {
    mul_out_npu_nocheck(result_cast, self_cast, other_cast);
  }

  if (result_type != calculate_type) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
    result.copy_(result_cast);
  }
}

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
  auto calculate_type = at::native::result_type(self, other);
  bool out_is_bool = (calculate_type == at::kBool);
  if (out_is_bool) {
    calculate_type = at::kFloat;
  }

  at::Tensor self_cast = (self.scalar_type() == calculate_type) ? self : self.to(calculate_type);
  at::Tensor other_cast = (other.scalar_type() == calculate_type) ? other : other.to(calculate_type);

  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self_cast) || npu_preparation::IsCPUScalar(self_cast);
  at::Tensor output_tensor = is_self_wrapped ? other_cast : self_cast;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_cast, other_cast);
  at::Tensor result = npu_preparation::apply_tensor(output_tensor, output_size);

  mul_out_npu_nocheck(result, self_cast, other_cast);
  if (out_is_bool) {
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor mul(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  mul_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) {
  return acl_op::mul_out(self, other, self);
}

at::Tensor& mul_(at::Tensor& self, const at::Scalar& other) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    mul_out_npu_nocheck(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    mul_out_npu_nocheck(self, self, other);
  }
  return self;
}
} // namespace acl_op
