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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor zeros_symint(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnInplaceZero, acl_op::zeros_symint(size, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  at::TensorOptions option = option.dtype(dtype_opt)
                                  .layout(layout_opt)
                                  .device(device_opt)
                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::apply_tensor_without_format(c10::asIntArrayRefUnchecked(size), option);
  return result.zero_();
}


at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnInplaceZero, acl_op::zeros(size, names, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  at::TensorOptions option = option.dtype(dtype_opt)
                                  .layout(layout_opt)
                                  .device(device_opt)
                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::apply_tensor_without_format(size, option);
  return result.zero_();
}

} // namespace op_api
