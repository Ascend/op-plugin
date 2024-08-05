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

at::Tensor& tanh_out(const at::Tensor& self, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnTanh, acl_op::tanh_out(self, result));
    TORCH_CHECK(!isIntegralType(result.scalar_type(), true), "result dtype can't be cast to the desired output type.\n",
                OPS_ERROR(ErrCode::TYPE));
    npu_preparation::check_tensor({self}, result, result, self.sizes());
    at_npu::native::OpPreparation::check_memory({self}, {result});
    EXEC_NPU_CMD(aclnnTanh, self, result);
    return result;
}

at::Tensor tanh(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnTanh, acl_op::tanh(self));
  auto output_dtype = self.dtype();
  if (isIntegralType(self.scalar_type(), true)) {
    output_dtype = at::kFloat;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(output_dtype));
  EXEC_NPU_CMD(aclnnTanh, self, result);
  return result;
}

at::Tensor& tanh_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceTanh, acl_op::tanh_(self));
  EXEC_NPU_CMD(aclnnInplaceTanh, self);
  return self;
}
}
