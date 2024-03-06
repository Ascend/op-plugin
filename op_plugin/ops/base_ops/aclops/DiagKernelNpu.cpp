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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> diag_npu_output_size(
    const at::Tensor& self,
    int64_t diagonal) {
  c10::SmallVector<int64_t, SIZE> shape;
  if (self.dim() == 1) {
    shape.emplace_back(self.size(0) + diagonal);
    shape.emplace_back(self.size(0) + diagonal);
    return shape;
  }
  int64_t m = self.size(0);
  int64_t n = self.size(1);
  if (m == n) {
    shape.emplace_back(m - diagonal);
  } else if (m < n) {
    shape.emplace_back(diagonal <= n - m ? m : n - diagonal);
  } else {
    shape.emplace_back(n - diagonal);
  }
  return shape;
}

at::Tensor& diag_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t diagonal) {
  at_npu::native::OpCommand cmd;
  if (self.dim() == 1) {
    cmd.Name("Diag");
  } else {
    cmd.Name("DiagPart");
  }
  cmd.Input(self)
      .Output(result)
      .Attr("diagonal", diagonal)
      .Run();
  return result;
}
} // namespace

at::Tensor& diag_out(
    const at::Tensor& self,
    int64_t diagonal,
    at::Tensor& result) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
      "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim(),
      OPS_ERROR(ErrCode::PARAM));
  diagonal = op_infer::make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
      "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal,
      OPS_ERROR(ErrCode::PARAM));

  auto output_size = diag_npu_output_size(self, diagonal);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    diag_out_npu_nocheck(contiguous_result, self, diagonal);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    diag_out_npu_nocheck(result, self, diagonal);
  }
  return result;
}

at::Tensor diag(
    const at::Tensor& self,
    int64_t diagonal) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
      "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim(),
      OPS_ERROR(ErrCode::PARAM));
  diagonal = op_infer::make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
      "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal,
      OPS_ERROR(ErrCode::PARAM));

  auto output_size = diag_npu_output_size(self, diagonal);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  diag_out_npu_nocheck(result, self, diagonal);
  return result;
}
} // namespace acl_op
