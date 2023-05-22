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

const at::Tensor& _conv_depthwise2d_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    const at::Tensor& result) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  const at::Tensor& weight_modify = weight.permute({1, 0, 2, 3});

  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  at::Tensor temp_out = result;

  at_npu::native::OpCommand cmd;
  cmd.Name("DepthwiseConv2D")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(weight_modify, "filter", ACL_FORMAT_NCHW);
  if (bias.defined()) {
      cmd.Input(bias);
  }
  cmd.Output(temp_out, "y", ACL_FORMAT_NCHW)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
  return result;
}

at::Tensor _conv_depthwise2d(
    const at::Tensor& self,
    const at::Tensor& weight,
    c10::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation) {
  int64_t N = self.size(0);
  int64_t Co = weight.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
  int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
  c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Ho, Wo};
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(self, output_size, ACL_FORMAT_NC1HWC0);
  return op_plugin::_conv_depthwise2d_out(self, weight, kernel_size, bias_opt, stride, padding, dilation, result);
}
} // namespace op_plugin
