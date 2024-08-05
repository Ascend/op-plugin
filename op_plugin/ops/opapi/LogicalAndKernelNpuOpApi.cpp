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

at::Tensor& logical_and_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLogicalAnd, acl_op::logical_and_out(self, other, result));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self, other}, result, output_size);
  EXEC_NPU_CMD(aclnnLogicalAnd, self, other, result);
  return result;
}

at::Tensor logical_and(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnLogicalAnd, acl_op::logical_and(self, other));
  auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLogicalAnd, self, other, result);
  return result;
}

at::Tensor& logical_and_(at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnLogicalAnd, acl_op::logical_and_(self, other));
  EXEC_NPU_CMD(aclnnInplaceLogicalAnd, self, other);
  return self;
}

}
