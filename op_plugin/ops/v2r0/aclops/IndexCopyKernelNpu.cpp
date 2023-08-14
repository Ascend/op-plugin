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
#include <ATen/native/NonSymbolicBC.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace op_plugin {
using npu_utils = at_npu::native::NpuUtils;

at::Tensor index_copy(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor contiguous_self(self.clone());
  if (!npu_utils::check_match(&self)) {
    contiguous_self = npu_utils::format_contiguous(contiguous_self);
  }
  return index_copy_npu_impl(dim, index, source, contiguous_self);

}

at::Tensor& index_copy_(
    at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor contiguous_self(self);
  if (!npu_utils::check_match(&self)) {
    contiguous_self = npu_utils::format_contiguous(self);
  }
  at::Tensor result = index_copy_npu_impl(dim, index, source, contiguous_self);
  npu_utils::format_fresh_view(self, result);

  return self;
}

} // namespace op_plugin
