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
at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);
  if(self.scalar_type() == at::kLong) {
    TORCH_NPU_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
        "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("NotEqual")
      .Expect(unified_result)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  if(self.scalar_type() == at::kLong) {
    TORCH_NPU_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
        "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("NotEqual")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor& ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  at::Tensor format_cast_of_other = npu_preparation::cast_to_ori_format(other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      calcu_op_util::GetTensorNpuFormat(format_cast_of_self),
      at::ScalarType::Bool,
      at::IntArrayRef(output_size));

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    ne_out_npu_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    ne_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
  }
  return result;
}

at::Tensor& ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  npu_preparation::CheckOut(
      {self},
      result,
      calcu_op_util::GetTensorNpuFormat(format_cast_of_self),
      at::ScalarType::Bool,
      format_cast_of_self.sizes());

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    ne_out_npu_nocheck(contiguous_result, format_cast_of_self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    ne_out_npu_nocheck(result, format_cast_of_self, other);
  }
  return result;
}

at::Tensor ne(const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return acl_op::ne(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return acl_op::ne(other, self.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
    at::Tensor format_cast_of_other = npu_preparation::cast_to_ori_format(other);

    auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);
    at::Tensor result = npu_preparation::apply_tensor(
        output_size,
        format_cast_of_self.options().dtype(at::kBool),
        format_cast_of_self);

    ne_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor ne(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);

  at::Tensor result = npu_preparation::apply_tensor(
      format_cast_of_self,
      format_cast_of_self.options().dtype(at::kBool));

  ne_out_npu_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor& ne_(at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return acl_op::ne_(self, other.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    npu_preparation::cast_to_ori_format(self);
    npu_preparation::cast_to_ori_format(other);
    npu_preparation::CheckMemory({self, other}, {self});

    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(at::ScalarType::Byte));

    if (!npu_utils::check_match(&self)) {
      at::Tensor contiguous_self = npu_utils::format_contiguous(self);
      ne_out_npu_nocheck(result, contiguous_self, other);
    } else {
      ne_out_npu_nocheck(result, self, other);
    }

    self.copy_(result);
    return self;
  }
}

at::Tensor& ne_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::cast_to_ori_format(self);
  at::Tensor result = npu_preparation::apply_tensor(
      self,
      self.options().dtype(at::ScalarType::Byte));

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    ne_out_npu_nocheck(result, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    ne_out_npu_nocheck(result, self, other);
  }

  self.copy_(result);
  return self;
}
} // namespace acl_op
