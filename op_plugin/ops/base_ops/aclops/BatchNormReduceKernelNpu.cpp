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

std::tuple<at::Tensor, at::Tensor> batch_norm_reduce(const at::Tensor& self, double eps) {
  auto output_size = {self.size(1)};
  at::Tensor sum = npu_preparation::ApplyTensor(output_size, self.options().dtype(at::kFloat), self);
  at::Tensor square_sum = npu_preparation::ApplyTensor(output_size, self.options().dtype(at::kFloat), self);

  at::Tensor self_copy = self;
  if (self.scalar_type() != at::kFloat) {
    self_copy = op_plugin::npu_dtype_cast(self_copy, at::kFloat);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("BNTrainingReduce")
      .Input(self_copy)
      .Output(sum)
      .Output(square_sum)
      .Run();

  return std::tie(sum, square_sum);
}
} // namespace op_plugin