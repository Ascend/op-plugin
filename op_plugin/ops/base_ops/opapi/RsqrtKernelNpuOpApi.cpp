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

at::Tensor& rsqrt_out(const at::Tensor& self, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnRsqrt, acl_op::rsqrt_out(self, result));
  auto result_dtype = self.scalar_type();
  if (isIntegralType(self.scalar_type(), true)) {
    result_dtype = at::kFloat;
  }
  TORCH_CHECK(!isIntegralType(result.scalar_type(), true),
              "result dtype ", result_dtype, " can't be cast to the desired output type ", result.dtype(), ".\n");
  npu_preparation::check_tensor({self}, result, result.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnRsqrt, self, result);
  return result;
}

at::Tensor& rsqrt_(at::Tensor& self)
{
  DO_COMPATIBILITY(aclnnInplaceRsqrt, acl_op::rsqrt_(self));
  TORCH_CHECK(!isIntegralType(self.scalar_type(), true),
              "result dtype float can't be cast to the desired output type ", self.dtype(), ".\n");
  EXEC_NPU_CMD(aclnnInplaceRsqrt, self);
  return self;
}

at::Tensor rsqrt(const at::Tensor& self)
{
  DO_COMPATIBILITY(aclnnRsqrt, acl_op::rsqrt(self));
  auto outDtype = self.dtype();
  if (isIntegralType(self.scalar_type(), true)) {
    outDtype = at::kFloat;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(outDtype));
  EXEC_NPU_CMD(aclnnRsqrt, self, result);
  return result;
}
}  // namespace op_api

