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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& gather_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  if (self.scalar_type() == at::kLong) {
    TORCH_NPU_WARN_ONCE("The oprator of gather is executed, Currently High Accuracy but Low Performance OP"
      "with 64-bit has been used,Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("GatherElements")
      .Input(self)
      .Input(index)
      .Attr("dim", dim)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  auto output_size = index.sizes();
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    gather_out_npu_nocheck(contiguous_result, self, dim, index, sparse_grad);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    gather_out_npu_nocheck(result, self, dim, index, sparse_grad);
  }
  return result;
}

at::Tensor& gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& result) {
  return op_plugin::gather_out(self, dimname_to_position(self, dim), index, sparse_grad, result);
}

at::Tensor gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  at::Tensor result = npu_preparation::ApplyTensor(self, index.sizes());
  gather_out_npu_nocheck(result, self, dim, index, sparse_grad);
  return result;
}

at::Tensor gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad) {
  return op_plugin::gather(self, dimname_to_position(self, dim), index, sparse_grad);
}
}  // op_plugin
