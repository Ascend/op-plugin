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
using npu_preparation = at_npu::native::OpPreparation;

static c10::SmallVector<int64_t, SIZE> diag_npu_output_size_opapi(const at::Tensor& self, int64_t diagonal)
{
    c10::SmallVector<int64_t, SIZE> shape;
    if (self.dim() < 2) {
        shape.emplace_back(self.size(0) + std::abs(diagonal));
        shape.emplace_back(self.size(0) + std::abs(diagonal));
        return shape;
    }
    int64_t m = self.size(0);
    int64_t n = self.size(1);
    if (diagonal > 0) {
        shape.emplace_back(std::min(m, n - diagonal));
        // Judge whether the parameter is out of range
        TORCH_CHECK(diagonal <= n,
                    "If the value is 2-dimensional tensor, the diagonal shoule be less than shape.Diagonal is ",
                    diagonal, OPS_ERROR(ErrCode::VALUE));
    } else {
        shape.emplace_back(std::min(m + diagonal, n));
        // Judge whether the parameter is out of range
        TORCH_CHECK(-diagonal <= m,
                    "If the value is 2-dimensional tensor, the diagonal shoule be less than shape.Diagonal is ",
                    diagonal, OPS_ERROR(ErrCode::VALUE));
    }
    return shape;
}

at::Tensor& diag_out(const at::Tensor& self, int64_t diagonal, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnDiag, acl_op::diag_out(self, diagonal, result));
  auto outputSize = diag_npu_output_size_opapi(self, diagonal);
  npu_preparation::check_tensor({self}, result, self.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnDiag, self, diagonal, result);
  return result;
}

at::Tensor diag(const at::Tensor& self, int64_t diagonal) {
  DO_COMPATIBILITY(aclnnDiag, acl_op::diag(self, diagonal));
  auto outputSize = diag_npu_output_size_opapi(self, diagonal);
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
  EXEC_NPU_CMD(aclnnDiag, self, diagonal, result);
  return result;
}

}  // namespace op_api