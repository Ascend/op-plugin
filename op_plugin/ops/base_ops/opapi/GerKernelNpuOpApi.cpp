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

at::Tensor& ger_out(const at::Tensor& self , const at::Tensor& vec2, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGer, acl_op::ger_out(self, vec2, result));
  auto output_size = op_infer::ger_output_size(self, vec2);
  auto result_type = at::result_type(self, vec2);
  npu_preparation::check_tensor({self, vec2}, result, result_type, output_size);

  // ger_out does not have a grad_fn, so it calls aclnnGer.
  EXEC_NPU_CMD(aclnnGer, self, vec2, result);
  return result;
}

at::Tensor ger(const at::Tensor& self, const at::Tensor& vec2) {
  TORCH_CHECK(self.dim() == 1, "Input1 must have only 1 dims.");
  TORCH_CHECK(vec2.dim() == 1, "Input2 must have only 1 dims.");

  // if ger calls aclnnGer, self does not have a grad_fn.
  // So ger is consistent with the torch, calls mul.
  return self.reshape({self.size(0), 1}) * vec2;
}
} // namespace op_api
