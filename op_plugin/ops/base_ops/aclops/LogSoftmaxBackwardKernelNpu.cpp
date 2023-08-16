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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
at::Tensor& log_softmax_backward_data_out_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  c10::SmallVector<int64_t, N> dim_list = {dim};
  at_npu::native::OpCommand cmd;
  cmd.Name("LogSoftmaxGrad")
      .Input(grad_output)
      .Input(output)
      .Output(result)
      .Attr("axis", dim_list)
      .Run();
  return result;
}
}

at::Tensor& _log_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    c10::ScalarType input_dtype,
    at::Tensor& result) {
  auto output_size = op_infer::input_same_output_size(grad_output);
  npu_preparation::CheckOut(
      {grad_output, output},
      result,
      grad_output,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contig_result = npu_utils::format_contiguous(result);
    log_softmax_backward_data_out_nocheck(contig_result, grad_output, output, dim, input_dtype);
    npu_utils::format_fresh_view(result, contig_result);
  } else {
    log_softmax_backward_data_out_nocheck(result, grad_output, output, dim, input_dtype);
  }
  return result;
}

at::Tensor _log_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  auto output_size = op_infer::input_same_output_size(grad_output);

  at::Tensor temp_output = output;
  if (calcu_op_util::GetTensorNpuFormat(temp_output) == ACL_FORMAT_NC1HWC0) {
    at_npu::native::NPUNativeFunctions::npu_format_cast(temp_output, calcu_op_util::GetTensorNpuFormat(grad_output));
  }
  at::Tensor grad_input = npu_preparation::apply_tensor(temp_output, output_size);
  log_softmax_backward_data_out_nocheck(grad_input, grad_output, temp_output, dim, input_dtype);
  return grad_input;
}
} // namespace acl_op
