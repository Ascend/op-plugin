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
at::Tensor rsub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnRsub, acl_op::rsub(self, other, alpha));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                           self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRsub, self, other, alpha, result);
  return result;
}

at::Tensor rsub(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnRsubs, acl_op::rsub(self, other, alpha));
  auto output_size = op_infer::input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                           self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRsubs, self, other, alpha, result);
  return result;
}

} // namespace op_api
