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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& linspace_npu_out_nocheck(
    at::Tensor& result,
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps) {
  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    acl_op::fill_(result, start);
  } else {
    c10::SmallVector<int64_t, N> size_vec = {steps};
    at_npu::native::OpCommand cmd;
    cmd.Name("LinSpace")
        .Input(start, at::ScalarType::Float)
        .Input(end, at::ScalarType::Float)
        .Input(size_vec, at::ScalarType::Int)
        .Output(result)
        .Run();
  }
  return result;
}
}  // namespace

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  bool result_is_not_float = (result.dtype() != at::kFloat) ? true : false;

  at::Tensor result_cast = result;
  if (result_is_not_float) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat);
  }

  if (!npu_utils::check_match(&result_cast)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
    linspace_npu_out_nocheck(contiguous_result, start, end, steps);
    npu_utils::format_fresh_view(result_cast, contiguous_result);
  } else {
    linspace_npu_out_nocheck(result_cast, start, end, steps);
  }

  if (result_is_not_float) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result.scalar_type());
    result.copy_(result_cast);
  } else {
    result = result_cast;
  }

  return result;
}

at::Tensor linspace(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt).layout(layout_opt).device(device).pinned_memory(pin_memory_opt);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat({steps}, option, ACL_FORMAT_ND);
  at::Tensor result_cast = result;

  bool result_is_not_float = (result.dtype() != at::kFloat) ? true : false;

  if (result_is_not_float) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat);
  }

  linspace_npu_out_nocheck(result_cast, start, end, steps);

  if (result_is_not_float) {
    result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result.scalar_type());
  }
  result = result_cast;
  return result;
}
}  // namespace acl_op
