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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor cumsum(
    const at::Tensor& self,
    int64_t dim,
    const c10::optional<at::ScalarType> dtype) {
  at::Tensor result;
  if (dtype.has_value()) {
    result = npu_preparation::apply_tensor(self, self.options().dtype(dtype.value()));
  } else if (self.scalar_type() == at::ScalarType::Bool) {
    result = npu_preparation::apply_tensor(self, self.options().dtype(at::kLong));
  } else {
    result = npu_preparation::apply_tensor(self);
  }
  return acl_op::cumsum_out(self, dim, dtype, result);
}
} // namespace acl_op
