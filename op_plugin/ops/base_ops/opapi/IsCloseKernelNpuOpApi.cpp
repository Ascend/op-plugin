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

at::Tensor isclose(const at::Tensor& self, const at::Tensor& other, double rtol, double atol, bool equal_nan) {
  DO_COMPATIBILITY(aclnnIsClose, acl_op::isclose(self, other, rtol, atol, equal_nan));

  // calculate the output size
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  if (at::isFloatingType(self.scalar_type()) && equal_nan) {
    output_size = self.sizes();
  }
  auto out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kBool));

  // construct the output tensor of the NPU
  EXEC_NPU_CMD(aclnnIsClose, self, other, rtol, atol, equal_nan, out);
  return out;
}
} // namespace op_api
