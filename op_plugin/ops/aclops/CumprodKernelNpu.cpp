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
  return result;
}

at::Tensor& cumprod_out(
    const at::Tensor& self,
    at::Dimname dim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  return acl_op::cumprod_out(self, dimname_to_position(self, dim), dtype, result);
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
      "." + OPS_ERROR(ErrCode::TYPE));
  return acl_op::cumprod_out(self, dim, dtype, self);
}

at::Tensor& cumprod_(at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  return acl_op::cumprod_(self, dimname_to_position(self, dim), dtype);
}

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor cumprod(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  at::Tensor self_cast = dtype.has_value() ? at_npu::native::custom_ops::npu_dtype_cast(self, dtype.value()) : self;
  at::Tensor result = npu_preparation::apply_tensor(self_cast);
  cumprod_out_nocheck(result, self_cast, dim);
  return result;
}
#endif

} // namespace acl_op
