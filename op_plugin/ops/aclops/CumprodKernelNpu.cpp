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
using npu_utils = at_npu::native::NpuUtils;

namespace {
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

at::Tensor& cumprod_out(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
    at::ScalarType dst_type = self.scalar_type();
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (result.defined()) {
        dst_type = result.scalar_type();
    }

    at::Tensor self_cp = self.scalar_type() == dst_type ? self :
        at_npu::native::custom_ops::npu_dtype_cast(self, dst_type);
    npu_preparation::CheckOut(
        {self_cp},
        result,
        npu_preparation::get_tensor_npu_format(result),
        dst_type,
        self_cp.sizes());
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        cumprod_out_nocheck(contiguous_result, self_cp, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        cumprod_out_nocheck(result, self_cp, dim);
    }
    at::namedinference::propagate_names(result, self);
    return result;
}
} // namespace acl_op
