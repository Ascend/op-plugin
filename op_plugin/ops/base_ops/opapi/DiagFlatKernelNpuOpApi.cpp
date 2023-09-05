// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

c10::SmallVector<int64_t, SIZE> diag_npu_output_size(const at::Tensor& self, int64_t offset) {
  c10::SmallVector<int64_t, SIZE> shape;
  if (self.dim() == 1) {
    shape.emplace_back(self.size(0) + std::abs(offset));
    shape.emplace_back(self.size(0) + std::abs(offset));
    return shape;
  }
  
  if (self.dim() > 1) {
    int64_t total_dim = 1;
    for (int i = 0; i < self.dim(); i++) {
      total_dim *= self.size(i);
    }
    shape.emplace_back(total_dim + std::abs(offset));
    shape.emplace_back(total_dim + std::abs(offset));
  }
  return shape;
}

at::Tensor diagflat(const at::Tensor& self, int64_t offset) {
  DO_COMPATIBILITY(aclnnDiagFlat, acl_op::diagflat(self, offset));
  // int64_t offset_s = static_cast<int64_t>(offset);
  auto output_size = diag_npu_output_size(self, offset);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnDiagFlat, self, offset, result);
  return result;
}

} // namespace at_npu
