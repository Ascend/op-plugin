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

at::Tensor& ger_out(const at::Tensor& self, const at::Tensor& vec2, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGer, acl_op::ger_out(self, vec2, result));
  auto output_size = op_infer::ger_output_size(self, vec2);
  auto result_type = at::result_type(self, vec2);
  npu_preparation::check_tensor({self, vec2}, result, result_type, output_size);

  EXEC_NPU_CMD(aclnnGer, self, vec2, result);
  return result;
}

at::Tensor ger(const at::Tensor& self, const at::Tensor& vec2) {
  DO_COMPATIBILITY(aclnnGer, acl_op::ger(self, vec2));
  auto output_size = op_infer::ger_output_size(self, vec2);
  auto result_type = at::result_type(self, vec2);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));

  EXEC_NPU_CMD(aclnnGer, self, vec2, result);
  return result;
}
} // namespace op_api
