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

at::Tensor& trunc_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnTrunc, acl_op::trunc_out(self, result));
  auto outputSize = self.sizes();
  npu_preparation::check_tensor({self}, result, self.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnTrunc, self, result);
  return result;
}

at::Tensor& trunc_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceTrunc, acl_op::trunc_(self));
  EXEC_NPU_CMD(aclnnInplaceTrunc, self);
  return self;
}

at::Tensor trunc(const at::Tensor& self) {
  auto outputSize = self.sizes();
  DO_COMPATIBILITY(aclnnTrunc, acl_op::trunc(self));
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
  EXEC_NPU_CMD(aclnnTrunc, self, result);
  return result;
}

} // namespace at_npu

