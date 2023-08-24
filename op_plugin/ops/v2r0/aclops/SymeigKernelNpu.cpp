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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

std::tuple<at::Tensor, at::Tensor> _symeig_helper(
    const at::Tensor& self,
    bool eigenvectors,
    bool upper) {
  auto self_sizes = self.sizes().vec();
  self_sizes.pop_back();
  auto eigvals = at::empty(self_sizes, self.options());

  if (self.numel() == 0) {
    return std::tuple<at::Tensor, at::Tensor>(eigvals, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  auto self_working_copy = self.clone();
  at_npu::native::OpCommand cmd;
  cmd.Name("SelfAdjointEig")
      .Input(self)
      .Output(eigvals)
      .Output(self_working_copy)
      .Attr("compute_v", true)
      .Run();

  if (eigenvectors) {
    return std::tuple<at::Tensor, at::Tensor>(eigvals, self_working_copy);
  } else {
    return std::tuple<at::Tensor, at::Tensor>(eigvals, at::empty({0}, self.options()));
  }
}

} // namespace acl_op
