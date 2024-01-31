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

at::Tensor& addcmul_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                        const at::Scalar& value, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAddcmul, acl_op::addcmul_out(self, tensor1, tensor2, value, result));
  auto mul_output_size = op_infer::broadcast_ops_npu_output_size(tensor1, tensor2);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self.sizes(), mul_output_size);
  npu_preparation::check_tensor({self}, result, self, output_size);
  EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                   const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnAddcmul, acl_op::addcmul(self, tensor1, tensor2, value));
  auto mul_output_size = op_infer::broadcast_ops_npu_output_size(tensor1, tensor2);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self.sizes(), mul_output_size);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor& addcmul_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceAddcmul, acl_op::addcmul_(self, tensor1, tensor2, value));
  EXEC_NPU_CMD(aclnnInplaceAddcmul, self, tensor1, tensor2, value);
  return self;
}

}  // namespace op_api
