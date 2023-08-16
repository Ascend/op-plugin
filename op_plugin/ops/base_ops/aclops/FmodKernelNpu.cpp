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

namespace {
at::Tensor& fmod_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("Mod")
      .Expect(unified_result)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& fmod_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar& other) {
  auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
  at_npu::native::OpCommand cmd;
  cmd.Name("Mod")
      .Expect(unified_result)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& fmod_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self, other},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    fmod_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    fmod_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor& fmod_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  npu_preparation::CheckOut({self}, result, self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    fmod_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    fmod_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor& fmod_(at::Tensor& self, const at::Scalar& other) {
  return acl_op::fmod_out(self, other, self);
}

at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other) {
  return acl_op::fmod_out(self, other, self);
}

at::Tensor fmod(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor fmod(const at::Tensor& self, const at::Tensor& other) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}
} // namespace acl_op
