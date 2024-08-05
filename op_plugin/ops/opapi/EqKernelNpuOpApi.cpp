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

at::Tensor& eq_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnEqTensor, acl_op::eq_out(self, other, result));

  if (npu_preparation::IsCPUScalar(other)) {
    return op_api::eq_out(self, other.item(), result);
  } else if (npu_preparation::IsCPUScalar(self)) {
    return op_api::eq_out(other, self.item(), result);
  }

  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self, other}, result, at::IntArrayRef(output_size));
  EXEC_NPU_CMD(aclnnEqTensor, self, other, result);
  return result;
}

at::Tensor eq(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnEqTensor, acl_op::eq(self, other));

  if (npu_preparation::IsCPUScalar(other)) {
    return op_api::eq(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    return op_api::eq(other, self.item());
  }

  // calculate the output size
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kBool));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnEqTensor, self, other, result);
  return result;
}

at::Tensor& eq_out_npu_scalar(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  EXEC_NPU_CMD(aclnnEqScalar, self, other, result);
  return result;
}

at::Tensor eq(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnEqScalar, acl_op::eq(self, other));

  // calculate the output size
  auto output_size = op_infer::input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result =
      npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kBool));

  // calculate the output result of the NPU
  eq_out_npu_scalar(result, self, other);
  return result;
}

at::Tensor& eq_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnEqScalar, acl_op::eq_out(self, other, result));
  npu_preparation::check_tensor({self}, result, self.sizes());

  eq_out_npu_scalar(result, self, other);

  return result;
}

at::Tensor& eq_(at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnInplaceEqTensor, acl_op::eq_(self, other));

  const std::initializer_list<at::Tensor> inputs = {self, other};
  const std::initializer_list<at::Tensor> outputs = {self};
  npu_preparation::check_memory(inputs, outputs);

  EXEC_NPU_CMD(aclnnInplaceEqTensor, self, other);
  return self;
}

at::Tensor& eq_(at::Tensor &self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceEqScalar, acl_op::eq_(self, other));

  EXEC_NPU_CMD(aclnnInplaceEqScalar, self, other);
  return self;
}
}
