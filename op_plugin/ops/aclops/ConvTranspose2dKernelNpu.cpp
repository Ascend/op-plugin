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

namespace {
at::Tensor& conv_transpose2d_out_npu(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> outputpadding = {0, 0, output_padding[0], output_padding[1]};
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string data_format = "NCHW";
  c10::SmallVector<int64_t, N> size_vector = op_infer::array_to_small_vector(result.sizes());

  at_npu::native::OpCommand cmd;
  cmd.Name("Conv2DTranspose")
      .Input(size_vector, at::kInt)
      .Input(input, "x")
      .Input(weight, "filter");
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(result, "y")
      .Attr("pads", paddings)
      .Attr("output_padding", outputpadding)
      .Attr("strides", strides_size)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", data_format)
      .Run();
  return result;
}
} // namespace

at::Tensor npu_conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  auto output_size = op_infer::conv_transpose2d_npu_output_size(
      input, weight, padding, output_padding, stride, dilation, groups);
  int64_t result_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
  at::Tensor result =
      npu_preparation::apply_tensor_with_format(output_size, input.options(), result_format);

  conv_transpose2d_out_npu(
      result, input, weight, bias, padding, output_padding, stride, dilation, groups);
  return result;
}
} // namespace acl_op
