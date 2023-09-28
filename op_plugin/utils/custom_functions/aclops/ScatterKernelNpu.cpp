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
at::Tensor& scatter_npu_common_nocheck(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ScatterElements")
      .Input(self)
      .Input(index)
      .Input(src)
      .Output(self)
      .Attr("axis", dim)
      .Run();
  return self;
}

at::Tensor& scatter_npu_src_impl(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Tensor& src_ex) {
  at::ScalarType self_type = self.scalar_type();
  if (self_type == at::ScalarType::Half) {
    self = acl_op::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::Tensor index(index_ex);
  if (index.scalar_type() == at::ScalarType::Half) {
    index = acl_op::npu_dtype_cast(index, at::ScalarType::Float);
  }

  at::Tensor src(src_ex);
  if (src.scalar_type() != self.scalar_type()) {
    src = acl_op::npu_dtype_cast(src, self.scalar_type());
  }

  scatter_npu_common_nocheck(self, dim, index, src);

  if (self.scalar_type() != self_type) {
    self = acl_op::npu_dtype_cast(self, self_type);
  }

  return self;
}
} // namespace acl_op
