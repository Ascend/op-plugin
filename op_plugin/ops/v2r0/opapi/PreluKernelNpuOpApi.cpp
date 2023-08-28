// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

at::Tensor _prelu_kernel(const at::Tensor& self, const at::Tensor& weight_) {
  DO_COMPATIBILITY(aclnnPrelu, acl_op::_prelu_kernel(self, weight_));
  // calculate the output size
  auto outputSize = op_infer::input_same_output_size(self);
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self, outputSize);

  EXEC_NPU_CMD(aclnnPrelu, self, weight_, result);
  return result;
}
}  // namespace op_api
