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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& eye_out_npu_nocheck(at::Tensor& result, int64_t n, int64_t m) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Eye")
      .Output(result)
      .Attr("num_rows", n)
      .Attr("num_columns", m)
      .Attr("dtype", result.scalar_type())
      .Run();

  return result;
}
} // namespace

at::Tensor& eye_out(int64_t n, int64_t m, at::Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if (m < 0) {
    m = n;
  }
  result.resize_({n, m});
  bool result_is_bool = result.scalar_type() == at::kBool;
  at::Tensor result_cp = result_is_bool ? op_plugin::npu_dtype_cast(result, at::kInt) : result;
  if (!npu_utils::check_match(&result_cp)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
    eye_out_npu_nocheck(contiguous_result, n, m);
    npu_utils::format_fresh_view(result_cp, contiguous_result);
  } else {
    eye_out_npu_nocheck(result_cp, n, m);
  }

  if (result_is_bool) {
    result_cp = op_plugin::npu_dtype_cast(result_cp, at::kBool);
    result.copy_(result_cp);
  }
  return result;
}

at::Tensor& eye_out(int64_t n, at::Tensor& result) {
  return op_plugin::eye_out(n, -1, result);
}

at::Tensor eye(
    int64_t n,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions option = option.dtype(dtype_opt).layout(layout_opt).device(device).pinned_memory(pin_memory_opt);

  c10::SmallVector<int64_t, N> output_size = {n, n};

  // The operator does not support the bool type and needs to be converted to an integer.
  at::Tensor result = (option.dtype() == at::kBool) ?
      npu_preparation::ApplyTensorWithFormat(output_size, option.dtype(at::kInt), ACL_FORMAT_ND) :
      npu_preparation::ApplyTensorWithFormat(output_size, option, ACL_FORMAT_ND);

  op_plugin::eye_out(n, result);
  if (option.dtype() == at::kBool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor eye(
    int64_t n,
    int64_t m,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
      .layout(layout_opt)
      .device(device)
      .pinned_memory(pin_memory_opt);

  // get the output size
  c10::SmallVector<int64_t, N> output_size = {n, m};

  // The operator does not support the bool type and needs to be converted to an integer.
  at::Tensor result = (option.dtype() == at::kBool) ?
      npu_preparation::ApplyTensorWithFormat(output_size, option.dtype(at::kInt), ACL_FORMAT_ND) :
      npu_preparation::ApplyTensorWithFormat(output_size, option, ACL_FORMAT_ND);

  eye_out_npu_nocheck(result, n, m);
  if (option.dtype() == at::kBool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }
  return result;
}
} // namespace op_plugin
