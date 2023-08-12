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
at::Tensor& sum_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& result) {
  return op_plugin::sum_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor sum(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  return op_plugin::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor sum(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
  return op_plugin::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}
} // namespace op_plugin
