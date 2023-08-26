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
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace{
at::Tensor& conv3d_backward_input_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  at::IntArrayRef input_size = input.sizes();
  at::Tensor weight_cast = at_npu::native::custom_ops::npu_dtype_cast(weight, grad.scalar_type());

  at_npu::native::OpCommand cmd;
  cmd.Name("Conv3DBackpropInput")
      .Input(input_size, at::kInt)
      .Input(weight_cast, "filter")
      .Input(grad, "out_backprop")
      .Output(grad_input, "y")
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", (string) "NCDHW")
      .Run();
  return grad_input;
}

at::Tensor& conv3d_backward_weight_nocheck(
    at::Tensor& grad_weight,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  at::IntArrayRef input_size = weight.sizes();

  at_npu::native::OpCommand cmd;
  cmd.Name("Conv3DBackpropFilter")
      .Input(input, "x")
      .Input(input_size, at::kInt)
      .Input(grad, "out_backprop")
      .Output(grad_weight, "y")
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", (string) "NCDHW")
      .Run();
  return grad_weight;
}

at::Tensor& conv3d_backward_bias_nocheck(
    at::Tensor& grad_bias,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  if (input.numel() == input.size(0) * input.size(1) * input.size(2)) {
    at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2)});
    acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0}, false, grad_view.scalar_type(), grad_bias);
  } else {
    at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1), grad.size(2), -1});
    acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2, 3}, false, grad_view.scalar_type(), grad_bias);
  }
  return grad_bias;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv3d_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation, 
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  if (grad_input_mask[0]) {
    // format should be NDC1HWC0
    grad_input = npu_preparation::ApplyTensorWithFormat(input, ACL_FORMAT_NDC1HWC0);
    conv3d_backward_input_nocheck(grad_input, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[1]) {
    // format should be FRACTAL_Z_3D
    grad_weight = npu_preparation::ApplyTensorWithFormat(
        weight.sizes(), weight.options().dtype(at::kFloat), calcu_op_util::GetTensorNpuFormat(weight));
    conv3d_backward_weight_nocheck(grad_weight, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[2]) {
    // format should be NCHW, gradias.size = grad.size(1)
    grad_bias = npu_preparation::ApplyTensorWithFormat({grad.size(1)}, grad.options(), ACL_FORMAT_NCHW);
    conv3d_backward_bias_nocheck(grad_bias, input, grad, weight, stride, padding, dilation, groups);
  }
  return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace acl_op
