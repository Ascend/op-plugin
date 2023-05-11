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
using calcu_op_util = at_npu::native::CalcuOpUtil;

std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor& self) {
  auto min = op_plugin::min(self);
  auto max = op_plugin::max(self);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> _aminmax(
    const at::Tensor& self,
    const int64_t dim,
    const bool keepdim) {
  auto min = op_plugin::min(self, {dim}, keepdim);
  auto max = op_plugin::max(self, {dim}, keepdim);
  return std::tie(std::get<0>(min), std::get<0>(max));
}

std::tuple<at::Tensor&, at::Tensor&> aminmax_out(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& min,
    at::Tensor& max) {
  if (dim.has_value()) {
    max = op_plugin::amax_out(self, dim.value(), keepdim, max);
    min = op_plugin::amin_out(self, dim.value(), keepdim, min);
  } else {
    at::SmallVector<int64_t, SIZE> dims = calcu_op_util::GetDimlistForTensor(self);
    max = op_plugin::amax_out(self, dims, keepdim, max);
    min = op_plugin::amin_out(self, dims, keepdim, min);
  }
  return std::tie(min,max);
}
} // namespace op_plugin
