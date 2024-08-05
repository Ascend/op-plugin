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

static void round_decimals_check(const at::Tensor& self, int64_t decimals)
{
    if (isIntegralType(self.scalar_type(), true)) {
        TORCH_CHECK(decimals == 0, "round_npu not implemented for ", toString(self.scalar_type()), " with decimals != 0",
                    OPS_ERROR(ErrCode::TYPE));
    }
}

at::Tensor& round_out(const at::Tensor& self, int64_t decimals, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnRoundDecimals, acl_op::round_out(self, decimals, result));
  round_decimals_check(self, decimals);
  npu_preparation::check_tensor({self}, result, self);
  EXEC_NPU_CMD(aclnnRoundDecimals, self, decimals, result);
  return result;
}

at::Tensor round(const at::Tensor& self, int64_t decimals) {
  DO_COMPATIBILITY(aclnnRoundDecimals, acl_op::round(self, decimals));
  round_decimals_check(self, decimals);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  EXEC_NPU_CMD(aclnnRoundDecimals, self, decimals, result);
  return result;
}

at::Tensor& round_(at::Tensor& self, int64_t decimals) {
  DO_COMPATIBILITY(aclnnInplaceRoundDecimals, acl_op::round_(self, decimals));
  round_decimals_check(self, decimals);
  EXEC_NPU_CMD(aclnnInplaceRoundDecimals, self, decimals);
  return self;
}
}
