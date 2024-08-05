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

at::Tensor sqrt(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSqrt, acl_op::sqrt(self));
  auto out_dtype = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(out_dtype));
  EXEC_NPU_CMD(aclnnSqrt, self, result);
  return result;
}

at::Tensor& sqrt_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSqrt, acl_op::sqrt_out(self, result));
  npu_preparation::check_tensor({self}, result, result, self.sizes());
  EXEC_NPU_CMD(aclnnSqrt, self, result);
  return result;
}

at::Tensor& sqrt_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceSqrt, acl_op::sqrt_(self));
  EXEC_NPU_CMD(aclnnInplaceSqrt, self);
  return self;
}

}
