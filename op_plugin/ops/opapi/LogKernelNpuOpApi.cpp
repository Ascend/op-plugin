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

at::Tensor& log_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLog, acl_op::log_out(self, result));
  if (!result.is_same(self)) {
    at::ScalarType expext_dtype = self.scalar_type();
    if (isIntegralType(self.scalar_type(), true)) {
      expext_dtype = at::kFloat;
    }
    if (isFloatingType(result.scalar_type()) ||
        isComplexType(result.scalar_type())) {
      expext_dtype = result.scalar_type();
    }
    at_npu::native::OpPreparation::check_tensor({self}, result, expext_dtype, self.sizes());
  }

  at_npu::native::OpPreparation::check_memory({self}, {result});
  EXEC_NPU_CMD(aclnnLog, self, result);
  return result;
}

at::Tensor log(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnLog, acl_op::log(self));
  // construct the output tensor of the NPU
  at::ScalarType expext_dtype = self.scalar_type();
  if (isIntegralType(self.scalar_type(), true)) {
    expext_dtype = at::kFloat;
  }
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(
      self.sizes(),
      self.options().dtype(expext_dtype));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLog, self, result);

  return result;
}

at::Tensor& log_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLog, acl_op::log_(self));
  EXEC_NPU_CMD(aclnnInplaceLog, self);

  return self;
}

}
