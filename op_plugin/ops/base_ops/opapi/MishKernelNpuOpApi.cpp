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
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& mish_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMish, acl_op::mish_out(self, result));
  npu_preparation::CheckOut({self}, result, self);
  
  EXEC_NPU_CMD(aclnnMish, self, result);
  return result;
}

at::Tensor mish(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnMish, acl_op::mish(self));
  at::Tensor result = npu_preparation::apply_tensor(self);
  EXEC_NPU_CMD(aclnnMish, self, result);
  return result;
}

at::Tensor& mish_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMish, acl_op::mish_(self));
  return op_api::mish_out(self, self);
}
}  // namespace op_api
