// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

namespace {
at::Tensor& logical_not_out_nocheck(at::Tensor& result, const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& logical_not_out(const at::Tensor& self, at::Tensor& result) {
  at::ScalarType result_dtype = result.scalar_type();
  at::ScalarType src_type = self.scalar_type();
  npu_preparation::CheckOut(
      {self},
      result,
      npu_preparation::get_tensor_npu_format(self),
      result_dtype,
      self.sizes());

  bool self_is_bool = src_type == at::ScalarType::Bool;
  bool result_is_bool = result_dtype == at::ScalarType::Bool;
  at::Tensor self_cast = self_is_bool ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool);
  at::Tensor result_cast = result_is_bool ? result : at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
  if (!npu_utils::check_match(&result_cast)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
    logical_not_out_nocheck(contiguous_result, self_cast);
    npu_utils::format_fresh_view(result_cast, contiguous_result);
  } else {
    logical_not_out_nocheck(result_cast, self_cast);
  }

  if (!result_is_bool) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_dtype);
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor logical_not(const at::Tensor& self) {
  at::Tensor self_cast =
      self.scalar_type() != at::ScalarType::Bool ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool) : self;
  at::Tensor result = npu_preparation::apply_tensor(self_cast);
  logical_not_out_nocheck(result, self_cast);
  return result;
}

at::Tensor& logical_not_(at::Tensor& self) {
  return acl_op::logical_not_out(self, self);
}
} // namespace acl_op
