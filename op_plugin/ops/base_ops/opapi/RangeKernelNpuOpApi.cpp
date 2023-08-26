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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& range_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnRange, acl_op::range_out(start, end, step, result));
  TORCH_CHECK(std::isfinite(start.toDouble()) && std::isfinite(end.toDouble()), "unsupported range: start -> end");

  float start_value = op_plugin::utils::get_scalar_float_value(start);
  float end_value = op_plugin::utils::get_scalar_float_value(end);
  float step_value = op_plugin::utils::get_scalar_float_value(step);
  
  TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero");
  TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
      "upper bound and larger bound inconsistent with step sign");
  TORCH_CHECK(isFloatingType(result.scalar_type()) || isIntegralType(result.scalar_type()),
              "out datatype: ", result.scalar_type(), " unsupported datatype");
  
  double output_size = 0;
  if (isFloatingType(result.scalar_type())) {
    output_size = std::floor((end.toDouble() - start.toDouble()) / step.toDouble());
  } else {
    output_size = std::floor(static_cast<double>((end.toLong() - start.toLong()) / step.toLong()));
  }
  output_size = static_cast<int64_t>(output_size) + 1;

  npu_preparation::check_tensor({ }, result, result.scalar_type(), result.sizes());

  if (result.numel() != output_size) {
    result.resize_({output_size});
  }

  EXEC_NPU_CMD(aclnnRange, start, end, step, result);
  return result;
}

}  // namespace op_api
