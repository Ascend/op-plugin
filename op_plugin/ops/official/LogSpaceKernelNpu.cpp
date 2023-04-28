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
at::Tensor& logspace_out_npu_nocheck(
    at::Tensor& result,
    at::Scalar start,
    at::Scalar end,
    int64_t steps,
    double base) {
  if (steps < 0) {
    TORCH_CHECK("please input steps > 0");
  }
  if (base <= 0) {
    printf("if base<=0, please input intenger start, end, (end-start)/(steps-1)");
  }
  at::Tensor inputs;
  if (result.scalar_type() == at::ScalarType::Half) {
    inputs = op_plugin::npu_dtype_cast(at::arange(0, steps, at::device(c10::DeviceType::PrivateUse1)), at::kHalf);
  } else if (result.scalar_type() == at::ScalarType::Float) {
    inputs = at::arange(0, steps, at::device(c10::DeviceType::PrivateUse1).dtype(at::kFloat));
  }

  int64_t dtype = 0;
  if (result.scalar_type() == at::ScalarType::Half) {
    dtype = 0;
  } else if (result.scalar_type() == at::ScalarType::Float) {
    dtype = 1;
  } else {
    TORCH_CHECK("only support float32 and float16");
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("LogSpaceD")
      .Input(inputs)
      .Output(result)
      .Attr("start", start)
      .Attr("end", end)
      .Attr("steps", steps)
      .Attr("base", static_cast<float>(base))
      .Attr("dtype", dtype)
      .Run();
  return result;
}
} // namespace

at::Tensor& logspace_out(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      { },
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      {steps});

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    logspace_out_npu_nocheck(contiguous_result, start, end, steps, base);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    logspace_out_npu_nocheck(result, start, end, steps, base);
  }
  return result;
}

at::Tensor logspace(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    double base,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = c10::device_or_default(device_opt);
  at::TensorOptions options;
  options = options.dtype(dtype_opt).layout(layout_opt).device(device).pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat({steps}, options, ACL_FORMAT_ND);
  return logspace_out_npu_nocheck(result, start, end, steps, base);
}
} // namespace op_plugin
