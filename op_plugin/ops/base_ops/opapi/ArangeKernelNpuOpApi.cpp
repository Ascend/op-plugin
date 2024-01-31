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

// bool inputs are considered integral
static inline bool all_integral(std::initializer_list<std::reference_wrapper<at::Scalar>> l) {
  for (at::Scalar& s : l) {
    if (!s.isIntegral(true)) {
      return false;
    }
  }
  return true;
}

static at::Tensor& arange_out_op_api(at::Scalar start, at::Scalar end, at::Scalar step, at::Tensor& result) {
  EXEC_NPU_CMD(aclnnArange, start, end, step, result);
  return result;
}

static int64_t get_result_size(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                               at::ScalarType resultType) {
  double size_arange = 0;
  // calculate the output size
  if (isFloatingType(resultType)) {
    if (step.toDouble() != 0) {
      size_arange = std::ceil(static_cast<double>(end.toDouble() - start.toDouble()) / step.toDouble());
    }
  } else {
    if (step.toLong() != 0) {
      size_arange = std::ceil(static_cast<double>(end.toLong() - start.toLong()) / step.toLong());
    }
  }
  return static_cast<int64_t>(size_arange);
}

at::Tensor arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                  c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                  c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange, acl_op::arange(start, end, step, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);

  at::Scalar start_opt = start;
  at::Scalar end_opt = end;
  at::Scalar step_opt = step;
  bool set_to_integral_dtype = !option.has_dtype() && all_integral({start_opt, end_opt, step_opt});
  if (set_to_integral_dtype) {
    option = option.dtype(at::ScalarType::Long);
  }

  int64_t size_value = get_result_size(start, end, step, c10::typeMetaToScalarType(option.dtype()));
  at::SmallVector<int64_t, op_infer::SIZE> outputSize = {size_value};
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, option);
  arange_out_op_api(start, end, step, result);
  return result;
}

at::Tensor arange(const at::Scalar& start, const at::Scalar& end, c10::optional<at::ScalarType> dtype_opt,
                  c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                  c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange, acl_op::arange(start, end, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  return op_api::arange(start, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor arange(const at::Scalar& end, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                  c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnArange, acl_op::arange(end, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  return op_api::arange(0, end, dtype_opt, layout_opt, device_opt, pin_memory_opt);  // start = 0
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnArange, acl_op::arange_out(start, end, step, result));

  int64_t size_value = get_result_size(start, end, step, result.scalar_type());
  at::SmallVector<int64_t, op_infer::SIZE> outputSize = {size_value};
  result.resize_(outputSize);
  arange_out_op_api(start, end, step, result);
  return result;
}

static at::Tensor& arange_start_end_out(at::Scalar start, at::Scalar end, at::Tensor& result) {
  at::Scalar step = 1;
  return op_api::arange_out(start, end, step, result);
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnArange, acl_op::arange_out(end, result));
  return arange_start_end_out(0, end, result);
}
}  // namespace op_api
