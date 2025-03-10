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

at::Tensor& eye_out(int64_t n, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnEye, acl_op::eye_out(n, result));
    TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n, OPS_ERROR(ErrCode::VALUE));
    result.resize_({n, n});
    EXEC_NPU_CMD(aclnnEye, n, n, result);
    return result;
}

at::Tensor& eye_out(int64_t n, int64_t m, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnEye, acl_op::eye_out(n, m, result));
    TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m, OPS_ERROR(ErrCode::VALUE));
    result.resize_({n, m});
    EXEC_NPU_CMD(aclnnEye, n, m, result);
    return result;
}

at::Tensor eye(
    int64_t n,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnEye, acl_op::eye(n, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  auto device = device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  // get the output size
  c10::SmallVector<int64_t, op_infer::N> output_size = {n, n};
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, option);

  EXEC_NPU_CMD(aclnnEye, n, n, result);

  return result;
}

at::Tensor eye(
    int64_t n,
    int64_t m,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnEye, acl_op::eye(n, m, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  auto device = device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  // get the output size
  c10::SmallVector<int64_t, op_infer::N> output_size = {n, m};
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, option);

  EXEC_NPU_CMD(aclnnEye, n, m, result);

  return result;
}
}
