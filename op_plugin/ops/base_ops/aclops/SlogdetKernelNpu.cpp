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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::tuple<at::Tensor&, at::Tensor&> slogdet_out_nocheck(
    at::Tensor& sign,
    at::Tensor& y,
    const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LogMatrixDeterminant")
      .Input(self)
      .Output(sign)
      .Output(y)
      .Run();

  return std::tie(sign, y);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> slogdet(const at::Tensor& self) {
  TORCH_CHECK(self.dim() >= 2, "input must be at least 2 dimensions");
  auto output_size = op_infer::array_to_small_vector(self.sizes());
  output_size.erase(output_size.end() - 2, output_size.end());
  at::Tensor sign = npu_preparation::apply_tensor(self, output_size);
  at::Tensor y = npu_preparation::apply_tensor(self, output_size);

  slogdet_out_nocheck(sign, y, self);

  return std::tie(sign, y);
}
} // namespace acl_op
