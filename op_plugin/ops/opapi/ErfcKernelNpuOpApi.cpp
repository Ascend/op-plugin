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

at::Tensor& erfc_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnErfc, acl_op::erfc_out(self, result));
  result.resize_(self.sizes());
  EXEC_NPU_CMD(aclnnErfc, self, result);
  return result;
}

at::Tensor& erfc_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceErfc, acl_op::erfc_(self));
  EXEC_NPU_CMD(aclnnInplaceErfc, self);
  return self;
}

at::Tensor erfc(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnErfc, acl_op::erfc(self));
  at::Tensor result;
  if (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long
      || self.scalar_type() == at::ScalarType::Int) {
    result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kFloat));
  } else {
    result = npu_preparation::apply_tensor_without_format(self);
  }
  EXEC_NPU_CMD(aclnnErfc, self, result);
  return result;
}
}
