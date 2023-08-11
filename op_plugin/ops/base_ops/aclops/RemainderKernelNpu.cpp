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

at::Tensor& remainder_out_scalar_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FloorMod")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& remainder_out_tensor_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0) {
    op_plugin::remainder_out(self, other.item(), result);
  } else {
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorMod")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}
} // namespace

at::Tensor& remainder_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& result) {
  npu_preparation::CheckOut({self}, result, self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    remainder_out_scalar_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    remainder_out_scalar_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor& remainder_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  at::Tensor output_tensor = calcu_op_util::IsScalarWrappedToTensor(self) ? other : self;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self},
      result,
      output_tensor,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    remainder_out_tensor_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    remainder_out_tensor_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor remainder(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  at::Tensor output_tensor = is_self_wrapped ? other : self;

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size,
      output_tensor.options(),
      calcu_op_util::GetTensorNpuFormat(output_tensor));

  remainder_out_tensor_npu_nocheck(result, self, other);

  return result;
}

at::Tensor remainder(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);

  remainder_out_scalar_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other) {
  return op_plugin::remainder_out(self, other, self);
}

at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other) {
  return op_plugin::remainder_out(self, other, self);
}

} // namespace op_plugin
