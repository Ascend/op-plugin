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
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor kl_div(const at::Tensor& self, const at::Tensor& target, int64_t reduction, bool log_target) {
  DO_COMPATIBILITY(aclnnKlDiv, acl_op::kl_div(self, target, reduction, log_target));
  at::IntArrayRef output_size;
  output_size = reduction == at::Reduction::None
                             ? op_infer::broadcast_ops_npu_output_size(self.sizes(), target.sizes())
                             : at::ArrayRef<int64_t>();
  at::ScalarType result_type = at::native::result_type(self, target);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnKlDiv, self, target, reduction, log_target, result);
  return result;
}

}
