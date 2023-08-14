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
at::Tensor& le_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LessEqual")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& le_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("LessEqual")
      .Expect(unified_result)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();

  return result;
}
}

at::Tensor& le_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  auto output_size = format_cast_of_self.sizes();
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    le_out_npu_nocheck(contiguous_result, format_cast_of_self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    le_out_npu_nocheck(result, format_cast_of_self, other);
  }
  return result;
}

at::Tensor& le_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  at::Tensor format_cast_of_other = npu_preparation::cast_to_ori_format(other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    le_out_npu_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    le_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
  }
  return result;
}

at::Tensor le(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      format_cast_of_self.sizes(),
      format_cast_of_self.options().dtype(at::kBool),
      ACL_FORMAT_ND);
  le_out_npu_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor le(const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return op_plugin::le(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return op_plugin::ge(other, self.item());
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

    le_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor& le_(at::Tensor& self, const at::Scalar& other) {
  npu_preparation::cast_to_ori_format(self);
  at::Tensor result = npu_preparation::apply_tensor(
      self,
      self.options().dtype(at::ScalarType::Byte));
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    le_out_npu_nocheck(result, contiguous_self, other);
  } else {
    le_out_npu_nocheck(result, self, other);
  }
  self.copy_(result);
  return self;
}

at::Tensor& le_(at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return op_plugin::le_(self, other.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    npu_preparation::cast_to_ori_format(self);
    at::Tensor ori_other = npu_preparation::cast_to_ori_format(other);
    npu_preparation::CheckMemory({self, ori_other}, {self});
    at::Tensor result = npu_preparation::apply_tensor(
        self,
        self.options().dtype(at::ScalarType::Byte));
    if (!npu_utils::check_match(&self)) {
      at::Tensor contiguous_self = npu_utils::format_contiguous(self);
      le_out_npu_nocheck(result, contiguous_self, ori_other);
    } else {
      le_out_npu_nocheck(result, self, ori_other);
    }
    self.copy_(result);
    return self;
  }
}
} // namespace op_plugin
