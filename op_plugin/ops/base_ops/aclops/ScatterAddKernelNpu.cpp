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
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& scatter_add_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  std::string reduction = "add";
  at_npu::native::OpCommand cmd;
  cmd.Name("ScatterElements")
      .Input(self)
      .Input(index)
      .Input(src)
      .Output(result)
      .Attr("axis", dim)
      .Attr("reduction", reduction)
      .Run();
  return result;
}
} // namespace

at::Tensor scatter_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  return self.clone(at::MemoryFormat::Contiguous).scatter_add_(dim, index, src);
}

at::Tensor& scatter_add_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  npu_preparation::CheckMemory({self, index, src}, {self});

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    scatter_add_out_npu_nocheck(contiguous_self, self, dim, index, src);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    scatter_add_out_npu_nocheck(self, self, dim, index, src);
  }

  return self;
}

at::Tensor scatter_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  return op_plugin::scatter_add(self, dimname_to_position(self, dim), index, src);
}

} // namespace op_plugin
