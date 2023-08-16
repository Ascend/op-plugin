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

at::Tensor& full_out(at::IntArrayRef size, const at::Scalar& fill_value, at::Tensor& result) {
  npu_preparation::CheckOut(
      {},
      result,
      result,
      size);
  acl_op::fill_(result, fill_value);
  return result;
}

at::Tensor full(
    at::IntArrayRef size,
    const at::Scalar& fill_value,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);
  if (!dtype_opt.has_value()) {
    if (fill_value.isBoolean()) {
      option = option.dtype(at::kBool);
    } else if (fill_value.isIntegral(false)) {
      option = option.dtype(at::kLong);
    } else {
      option = option.dtype(c10::get_default_dtype());
    }
  }
  at::Tensor result = npu_preparation::ApplyTensorWithSizes(size, option);
  acl_op::fill_(result, fill_value);
  return result;
}
} // namespace acl_op
