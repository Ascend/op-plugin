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

at::Tensor _log_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  DO_COMPATIBILITY(aclnnLogSoftmax, acl_op::_log_softmax(self, dim, half_to_float));
  // construct the output tensor of the NPU
  at::Tensor result;
  if (half_to_float) {
    result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(),
                                        self.options().dtype(c10::ScalarType::Float));
  } else {
    result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
  }

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLogSoftmax, self, dim, result);

  return result;
}

at::Tensor& _log_softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float,
                             at::Tensor& out) {
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLogSoftmax, self, dim, out);
  return out;
}

}
