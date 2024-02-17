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

namespace{
at::Tensor& cumprod_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim) {
  at::Tensor self_not_0d = (self.dim() == 0) ? self.unsqueeze(0) : self;
  at::Scalar axis = dim;
  at_npu::native::OpCommand cmd;
  cmd.Name("Cumprod")
      .Input(self_not_0d)
      .Input(axis, at::kLong)
      .Attr("exclusive", (bool)false)
      .Attr("reverse", (bool)false)
      .Output(result)
      .Run();
  result = (self.dim() == 0) ? result.squeeze(0) : result;
  return result;
}
} // namespace

at::Tensor cumprod(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  at::Tensor self_cast = dtype.has_value() ? at_npu::native::custom_ops::npu_dtype_cast(self, dtype.value()) : self;
  at::Tensor result = npu_preparation::apply_tensor(self_cast);
  cumprod_out_nocheck(result, self_cast, dim);
  return result;
}
} // namespace acl_op
