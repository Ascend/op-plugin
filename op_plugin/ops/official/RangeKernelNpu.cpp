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

at::Tensor& range_out_nocheck(
    at::Tensor& result,
    at::Scalar start,
    at::Scalar end,
    at::Scalar step) {
  // generate x assistant tensor
  int value = result.size(0);
  c10::SmallVector<int64_t, N> tmp_vector = {};
  for (int i = 0; i < value; i++) {
    tmp_vector.emplace_back(i);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("RangeD")
      .Input(tmp_vector, result.scalar_type())
      .Output(result)
      .Attr("start", start)
      .Attr("limit", end)
      .Attr("delta", step)
      .Run();

  return result;
}
} // namespace

at::Tensor range(
    const at::Scalar& start,
    const at::Scalar& end,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);
  return at::range(start, end, 1, option);
}

at::Tensor range(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);

  float start_value = op_plugin::utils::get_scalar_float_value(start);
  float end_value = op_plugin::utils::get_scalar_float_value(end);
  float step_value = op_plugin::utils::get_scalar_float_value(step);

  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  auto output_size = op_infer::range_npu_output_size(start_value, end_value, step_value);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(output_size, option, ACL_FORMAT_NCHW);
  return range_out_nocheck(result, start, end, step);
}

at::Tensor& range_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& result) {
  float start_value = op_plugin::utils::get_scalar_float_value(start);
  float end_value = op_plugin::utils::get_scalar_float_value(end);
  float step_value = op_plugin::utils::get_scalar_float_value(step);

  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");

  auto output_size = op_infer::range_npu_output_size(start_value, end_value, step_value);
  npu_preparation::CheckOut(
      { },
      result,
      ACL_FORMAT_NCHW,
      result.scalar_type(),
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    range_out_nocheck(contiguous_result, start, end, step);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    range_out_nocheck(result, start, end, step);
  }

  return result;
}

} // namespace op_plugin
