// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor bitwise_right_shift(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(
      other, self.scalar_type(), self.device());
  at::Tensor result = npu_preparation::apply_tensor(self);
  EXEC_NPU_CMD(aclnnRightShift, self, scalar_tensor, result);
  return result;
}

at::Tensor& bitwise_right_shift_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
  at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(
      other, self.scalar_type(), self.device());
  EXEC_NPU_CMD(aclnnRightShift, self, scalar_tensor, out);
  return out;
}

at::Tensor& bitwise_right_shift_(at::Tensor& self, const at::Scalar& other) {
  at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(
      other, self.scalar_type(), self.device());
  EXEC_NPU_CMD(aclnnRightShift, self, scalar_tensor, self);
  return self;
}

at::Tensor bitwise_right_shift(const at::Scalar& self, const at::Tensor& other) {
  at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(
      self, other.scalar_type(), other.device());
  at::Tensor result = npu_preparation::apply_tensor(other);
  EXEC_NPU_CMD(aclnnRightShift, scalar_tensor, other, result);
  return result;
}

} // namespace op_api
