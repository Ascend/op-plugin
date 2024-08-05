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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& bitwise_xor_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    const at::Scalar &other_value = other.item();
    EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other_value, result);
  } else if (npu_preparation::IsCPUScalar(self)) {
    const at::Scalar &self_value = self.item();
    EXEC_NPU_CMD(aclnnBitwiseXorScalar, other, self_value, result);
  } else {
    EXEC_NPU_CMD(aclnnBitwiseXorTensor, self, other, result);
  }

  return result;
}
} // namespace

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, acl_op::bitwise_xor_out(self, other, result));
  npu_preparation::check_tensor({self}, result, self.sizes());
  EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other, result);
  return result;
}

at::Tensor bitwise_xor(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, acl_op::bitwise_xor(self, other));

  at::Tensor result;
  if ((self.scalar_type() == at::ScalarType::Bool) && (!other.isBoolean())) {
    result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kLong));
  } else {
    result = npu_preparation::apply_tensor_without_format(self);
  }

  EXEC_NPU_CMD(aclnnBitwiseXorScalar, self, other, result);
  return result;
}

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, acl_op::bitwise_xor_out(self, other, result));
  DO_COMPATIBILITY(aclnnBitwiseXorTensor, acl_op::bitwise_xor_out(self, other, result));

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self}, result, output_size);

  bitwise_xor_out_nocheck(result, self, other);
  return result;
}

at::Tensor bitwise_xor(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnBitwiseXorScalar, acl_op::bitwise_xor(self, other));
  DO_COMPATIBILITY(aclnnBitwiseXorTensor, acl_op::bitwise_xor(self, other));

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));

  bitwise_xor_out_nocheck(result, self, other);
  return result;
}

at::Tensor& bitwise_xor_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceBitwiseXorTensor, acl_op::bitwise_xor_(self, other));
  npu_preparation::check_memory({self, other}, {self});
  EXEC_NPU_CMD(aclnnInplaceBitwiseXorTensor, self, other);
  return self;
}

at::Tensor& bitwise_xor_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceBitwiseXorScalar, acl_op::bitwise_xor_(self, other));
  EXEC_NPU_CMD(aclnnInplaceBitwiseXorScalar, self, other);
  return self;
}
}
