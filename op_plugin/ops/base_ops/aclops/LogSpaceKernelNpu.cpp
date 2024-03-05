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
at::Tensor& logspace_out_nocheck(
    at::Tensor& result,
    at::Scalar start,
    at::Scalar end,
    int64_t steps,
    double base) {
    TORCH_CHECK(steps >= 0, "logspace requires non-negative steps, given steps is ", steps,
        OPS_ERROR(ErrCode::PARAM));
    if ((base <= 0) && ((!start.isIntegral(false)) || (!end.isIntegral(false)))) {
        std::cout << "Warning: start and end in logspace should both be int when base <= 0, "
            << "get type " << start.type() << " and" << end.type() << std::endl;
    }

    at::Tensor inputs;
    int64_t dtype = 0;
    auto result_type = result.scalar_type();
    if (result_type == at::ScalarType::Half) {
        inputs = at_npu::native::custom_ops::npu_dtype_cast(
            at::arange(0, steps, at::device(torch_npu::utils::get_npu_device_type())),
            at::kHalf);
        dtype = 0;
    } else if (result_type == at::ScalarType::Float) {
        inputs = at::arange(0, steps, at::device(torch_npu::utils::get_npu_device_type()).dtype(at::kFloat));
        dtype = 1;
    } else {
        TORCH_CHECK(false, "logspace only support float32 and float16, given type is ", result_type,
            OPS_ERROR(ErrCode::TYPE));
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
    logspace_out_nocheck(contiguous_result, start, end, steps, base);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    logspace_out_nocheck(result, start, end, steps, base);
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
  at::TensorOptions options = c10::TensorOptions()
      .dtype(dtype_opt).layout(layout_opt).device(device).pinned_memory(pin_memory_opt);
  at::Tensor result = npu_preparation::apply_tensor_with_format({steps}, options, ACL_FORMAT_ND);
  return logspace_out_nocheck(result, start, end, steps, base);
}
} // namespace acl_op
