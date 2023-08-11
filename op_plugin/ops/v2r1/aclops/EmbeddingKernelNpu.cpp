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

namespace {
at::Tensor& embedding_out_nocheck(
    at::Tensor& result,
    const at::Tensor& weight,
    const at::Tensor& indices) {
  c10::SmallVector<int64_t, N> dim_vec = {0};
  int64_t batch_dims = 0;

  at_npu::native::OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(weight)
      .Input(indices)
      .Input(dim_vec)
      .Output(result)
      .Attr("batch_dims", batch_dims)
      .Run();

  return result;
}
} // namespace

at::Tensor embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  auto output_size = op_infer::array_to_small_vector(indices.sizes());
  output_size.emplace_back(weight.size(weight.dim() - 1));
  at::Tensor result = npu_preparation::ApplyTensor(weight, output_size);

  embedding_out_nocheck(result, weight, indices);
  return result;
}
} // namespace op_plugin
