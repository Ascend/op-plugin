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

namespace op_api{
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& xlogy_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYTensor, acl_op::xlogy_out(self, other, result));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self, other}, result, result, output_size);
  EXEC_NPU_CMD(aclnnXLogYTensor, self, other, result);
  return result;
}

at::Tensor& xlogy_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYScalarOther, acl_op::xlogy_out(self, other, result));
  npu_preparation::check_tensor({self}, result, result, self.sizes());
  EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, result);
  return result;
}

at::Tensor& xlogy_out(const at::Scalar& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnXLogYScalarSelf, acl_op::xlogy_out(self, other, result));
  npu_preparation::check_tensor({other}, result, result, other.sizes());
  EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, result);
  return result;
}

at::Tensor xlogy(const at::Tensor& self, const at::Tensor& other) {
    DO_COMPATIBILITY(aclnnXLogYTensor, acl_op::xlogy(self, other));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    auto result_type = at::result_type(self, other);
    result_type = (isIntegralType(result_type, true)) ? at::kFloat : result_type;
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    EXEC_NPU_CMD(aclnnXLogYTensor, self, other, result);
    return result;
}

at::Tensor xlogy(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnXLogYScalarOther, acl_op::xlogy(self, other));
  at::ScalarType result_type = at::result_type(self, other);
  result_type = (isIntegralType(result_type, true)) ? at::kFloat : result_type;
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, result);
  return result;
}

at::Tensor xlogy(const at::Scalar& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnXLogYScalarSelf, acl_op::xlogy(self, other));
  at::ScalarType result_type = at::result_type(self, other);
  result_type = (isIntegralType(result_type, true)) ? at::kFloat : result_type;  
  at::Tensor result = npu_preparation::apply_tensor_without_format(other.sizes(), other.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, result);
  return result;
}

at::Tensor& xlogy_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceXLogYTensor, acl_op::xlogy_(self, other));
  npu_preparation::CheckMemory({self, other}, {self});
  EXEC_NPU_CMD(aclnnInplaceXLogYTensor, self, other);
  return self;
}

at::Tensor& xlogy_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceXLogYScalarOther, acl_op::xlogy_(self, other));
  EXEC_NPU_CMD(aclnnInplaceXLogYScalarOther, self, other);
  return self;
}

} // namespace op_api

