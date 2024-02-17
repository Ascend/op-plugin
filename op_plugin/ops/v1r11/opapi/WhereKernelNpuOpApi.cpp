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

at::Tensor _s_where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::_s_where(condition, self, other));
  auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnSWhere, condition, self, other, result);
  return result;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, acl_op::where(condition, self, other));
  return at::_s_where(condition, self, other);
}

} // namespace op_api
