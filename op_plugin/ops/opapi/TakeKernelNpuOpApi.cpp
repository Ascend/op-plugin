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

at::Tensor& take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnTake, acl_op::take_out(self, index, result));
  npu_preparation::check_tensor({self, index}, result, self.scalar_type(), index.sizes());
  EXEC_NPU_CMD(aclnnTake, self, index, result);
  return result;
}

at::Tensor take(const at::Tensor& self, const at::Tensor& index) {
  DO_COMPATIBILITY(aclnnTake, acl_op::take(self, index));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, index.sizes());
  EXEC_NPU_CMD(aclnnTake, self, index, result);
  return result;
}

}
