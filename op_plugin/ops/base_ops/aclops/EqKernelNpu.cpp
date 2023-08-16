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
using npu_utils = at_npu::native::NpuUtils;

namespace{
at::Tensor& eq_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor self_cast = self.dtype() == at::kInt ? acl_op::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor other_cast = other.dtype() == at::kInt ? acl_op::npu_dtype_cast(other, at::kFloat) : other;
  auto unified_result = npu_preparation::comparison_op_check(result, self_cast, other_cast, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("Equal")
      .Expect(unified_result)
      .Input(self_cast)
      .Input(other_cast)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& eq_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar other) {
  at::Tensor self_cast = self.dtype() == at::kInt ? acl_op::npu_dtype_cast(self, at::kFloat) : self;
  at_npu::native::OpCommand cmd;
  cmd.Name("Equal")
      .Input(self_cast)
      .Input(other, self_cast.scalar_type())
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& eq_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    eq_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    eq_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor& eq_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      self.sizes());
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    eq_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    eq_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor eq(
    const at::Tensor& self,
    const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    return acl_op::eq(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return acl_op::eq(other, self.item());
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

    eq_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor eq(
    const at::Tensor& self,
    const at::Scalar& other) {
  at::Tensor format_cast_of_self = npu_preparation::cast_to_ori_format(self);

  at::Tensor result = npu_preparation::apply_tensor_with_format(
      format_cast_of_self.sizes(),
      format_cast_of_self.options().dtype(at::kBool),
      ACL_FORMAT_ND);

  eq_out_npu_nocheck(result, format_cast_of_self, other);
  return result;
}

at::Tensor& eq_(
    at::Tensor& self,
    const at::Tensor& other) {
  npu_preparation::cast_to_ori_format(self);
  return acl_op::eq_out(self, other, self);
}

at::Tensor& eq_(
    at::Tensor& self,
    const at::Scalar& other) {
  npu_preparation::cast_to_ori_format(self);
  return acl_op::eq_out(self, other, self);
}
} // namespace acl_op
