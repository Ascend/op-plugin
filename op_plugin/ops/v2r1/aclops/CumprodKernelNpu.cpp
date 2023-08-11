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

namespace{
at::Tensor& cumprod_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim) {
  at::Scalar axis = dim;
  at_npu::native::OpCommand cmd;
  cmd.Name("Cumprod")
      .Input(self)
      .Input(axis, at::kLong)
      .Attr("exclusive", (bool)false)
      .Attr("reverse", (bool)false)
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor& cumprod_out(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  at::ScalarType dst_type;
  if (dtype.has_value()) {
    dst_type = dtype.value();
  } else if (result.defined()) {
    dst_type = result.scalar_type();
  } else {
    dst_type = self.scalar_type();
  }

  at::Tensor self_cp = self.scalar_type() == dst_type ? self :
      op_plugin::npu_dtype_cast(self, dst_type);
  npu_preparation::CheckOut(
      {self_cp},
      result,
      self_cp);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    cumprod_out_nocheck(contiguous_result, self_cp, dim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    cumprod_out_nocheck(result, self_cp, dim);
  }
  return result;
}

at::Tensor& cumprod_out(
    const at::Tensor& self,
    at::Dimname dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  return op_plugin::cumprod_out(self, dimname_to_position(self, dim), dtype, result);
}

at::Tensor& cumprod_(
    at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value() || (self.scalar_type() == dtype.value()),
      "provided dtype must match the dtype of self tensor in cumprod. Got ",
      toString(self.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  return op_plugin::cumprod_out(self, dim, dtype, self);
}

at::Tensor& cumprod_(at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  return op_plugin::cumprod_(self, dimname_to_position(self, dim), dtype);
}
} // namespace op_plugin
