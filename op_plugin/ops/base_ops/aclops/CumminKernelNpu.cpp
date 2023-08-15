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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

void cummin_out_npu_nocheck (
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Cummin")
      .Input(self)
      .Output(values)
      .Output(indices)
      .Attr("axis", dim)
      .Run();
}

void _cummin_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  // process aicpu
  if (self.scalar_type() == at::kLong) {
    at::Tensor values_temp = npu_preparation::ApplyTensor(self);
    at::Tensor indices_temp = npu_preparation::ApplyTensor(self, self.options().dtype(at::kLong));
    cummin_out_npu_nocheck(values_temp, indices_temp, self, dim);
    values.copy_(values_temp);
    indices.copy_(indices_temp);
  } else {
    // process aicore
    int64_t first_dim = op_plugin::utils::make_warp_dim(0, self.dim());
    if (dim != first_dim) {
      c10::SmallVector<int64_t, SIZE> perm;
      for (int64_t i = 0; i < self.dim(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[dim], perm[first_dim]);

      at::Tensor transpose_self = op_plugin::npu_transpose(self, perm, true);
      auto output_size = op_infer::transpose_npu_output_size(values, perm);
      at::Tensor transpose_value = npu_preparation::ApplyTensor(self, output_size);
      at::Tensor transpose_indices = npu_preparation::ApplyTensor(output_size, self.options().dtype(at::kInt), self);

      cummin_out_npu_nocheck(transpose_value, transpose_indices, transpose_self, first_dim);
      // Indices must to be long
      transpose_indices = op_plugin::npu_dtype_cast(transpose_indices, at::kLong);
      op_plugin::npu_transpose_out(transpose_value, perm, true, values);
      op_plugin::npu_transpose_out(transpose_indices, perm, true, indices);
    } else {
      at::Tensor values_temp = npu_preparation::ApplyTensor(self);
      at::Tensor indices_temp = npu_preparation::ApplyTensor(self, self.options().dtype(at::kInt));
      cummin_out_npu_nocheck(values_temp, indices_temp, self, dim);
      indices_temp = op_plugin::npu_dtype_cast(indices_temp, at::kLong);
      values.copy_(values_temp);
      indices.copy_(indices_temp);
    }
  }
}
} // namespace op_plugin