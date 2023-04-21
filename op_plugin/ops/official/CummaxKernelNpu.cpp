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

void _cummax_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  at::Tensor values_temp = npu_preparation::ApplyTensor(self);
  at::Tensor indices_temp = npu_preparation::ApplyTensor(self, self.options().dtype(at::kLong));

  at_npu::native::OpCommand cmd;
  cmd.Name("Cummax")
      .Input(self)
      .Output(values_temp)
      .Output(indices_temp)
      .Attr("dim", dim)
      .Run();

  values.copy_(values_temp);
  indices.copy_(indices_temp);
}
} // namespace op_plugin
