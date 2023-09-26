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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& ones_out(at::IntArrayRef size, at::Tensor& result) {
  result.resize_(size);
  return acl_op::one_(result);
}

at::Tensor ones(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
      .layout(layout_opt)
      .device(device)
      .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::apply_tensor_with_format(size, option, ACL_FORMAT_ND);

  return acl_op::one_(result);
}

at::Tensor ones(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
      .layout(layout_opt)
      .device(device)
      .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::apply_tensor_with_format(size, option, ACL_FORMAT_ND);

  return acl_op::one_(result);
}
}  // namespace acl_op
