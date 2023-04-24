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

at::Tensor& ones_out(at::IntArrayRef size, at::Tensor& result) {
  result.resize_(size);
  return op_plugin::one_(result);
}

at::Tensor ones(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
      .layout(layout_opt)
      .device(device)
      .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);

  return op_plugin::one_(result);
}

at::Tensor ones(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
      .layout(layout_opt)
      .device(device)
      .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);

  return op_plugin::one_(result);
}
}  // namespace op_plugin