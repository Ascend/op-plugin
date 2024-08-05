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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// pow.Tensor_Tensor_out
at::Tensor& pow_out(const at::Tensor& self, const at::Tensor& exp, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnPowTensorTensor, acl_op::pow_out(self, exp, result));
  auto outputSize = op_infer::broadcast_ops_npu_output_size(self, exp);
  npu_preparation::check_tensor({self, exp}, result, result, outputSize);
  npu_preparation::check_memory({self, exp}, {result});

  EXEC_NPU_CMD(aclnnPowTensorTensor, self, exp, result);
  return result;
}

// pow.Tensor_Scalar_out
at::Tensor& pow_out(const at::Tensor& self, const at::Scalar& exp, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnPowTensorScalar, acl_op::pow_out(self, exp, result));
  auto resultType = at::result_type(self, exp);
  npu_preparation::check_tensor({self}, result, resultType, self.sizes());
  npu_preparation::check_memory({self}, {result});

  EXEC_NPU_CMD(aclnnPowTensorScalar, self, exp, result);
  return result;
}

// pow.Scalar_out
at::Tensor &pow_out(const at::Scalar& self, const at::Tensor &exp, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnPowScalarTensor, acl_op::pow_out(self, exp, result));
  npu_preparation::check_tensor({exp}, result, result.scalar_type(), exp.sizes());

  EXEC_NPU_CMD(aclnnPowScalarTensor, self, exp, result);
  return result;
}

at::Tensor pow(const at::Tensor& self, const at::Tensor& exp) {
  DO_COMPATIBILITY(aclnnPowTensorTensor, acl_op::pow(self, exp));
  // calculate the output size
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, exp);
  at::ScalarType result_type = at::result_type(self, exp);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));

  EXEC_NPU_CMD(aclnnPowTensorTensor, self, exp, result);
  return result;
}

at::Tensor pow(const at::Tensor& self, const at::Scalar& exp) {
  DO_COMPATIBILITY(aclnnPowTensorScalar, acl_op::pow(self, exp));
  auto outputSize = op_infer::input_same_output_size(self);
  auto resultType = at::result_type(self, exp);
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(resultType));

  EXEC_NPU_CMD(aclnnPowTensorScalar, self, exp, result);
  return result;
}

at::Tensor pow(const at::Scalar& self, const at::Tensor& exp) {
  DO_COMPATIBILITY(aclnnPowScalarTensor, acl_op::pow(self, exp));
  at::ScalarType result_type = at::result_type(self, exp);
  at::Tensor result = npu_preparation::apply_tensor_without_format(exp.sizes(), exp.options().dtype(result_type));

  EXEC_NPU_CMD(aclnnPowScalarTensor, self, exp, result);
  return result;
}

at::Tensor &pow_(at::Tensor &self, const at::Tensor &exp) {
  DO_COMPATIBILITY(aclnnInplacePowTensorTensor, acl_op::pow_(self, exp));
  EXEC_NPU_CMD(aclnnInplacePowTensorTensor, self, exp);
  return self;
}

at::Tensor &pow_(at::Tensor &self, const at::Scalar& exp) {
  DO_COMPATIBILITY(aclnnInplacePowTensorScalar, acl_op::pow_(self, exp));
  EXEC_NPU_CMD(aclnnInplacePowTensorScalar, self, exp);
  return self;
}
}
