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
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
bool is_special_conv1d(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  if (stride[1] > 63 &&
      stride[1] == weight.size(3) &&
      padding[1] == 0 &&
      dilation[1] == 1 &&
      groups == 1 &&
      input.size(1) == 1) {
    return true;
  } else {
    return false;
  }
}

at::Tensor& conv2d_backward_input_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  // support special scenario
  if (is_special_conv1d(input, weight, stride, padding, dilation, groups)) {
    at::Tensor mm_input = grad.permute({0, 2, 1});
    at::Tensor mm_other = weight.reshape({weight.size(0), weight.size(3)});
    at::Tensor mm_result = at::matmul(mm_input, mm_other);
    grad_input = mm_result.reshape({grad.size(0), 1, 1, grad.size(2) * weight.size(3)});
    return grad_input;
  }

  c10::SmallVector<int64_t, N> dim_list = op_infer::array_to_small_vector(input.sizes());
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";

  at_npu::native::OpCommand cmd;
  cmd.Name("Conv2DBackpropInput")
      .Input(dim_list, at::kInt)
      .Input(weight, "filter", ACL_FORMAT_NCHW)
      .Input(grad, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_input;
}

at::Tensor& conv2d_backward_weight_out_nocheck(
    at::Tensor& grad_weight,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  // support special scenario
  if (is_special_conv1d(input, weight, stride, padding, dilation, groups)) {
    at::Tensor mm_input = grad.permute({1, 0, 2}).reshape({grad.size(1), grad.size(0) * grad.size(2)});
    at::Tensor mm_other = input.reshape({input.size(0), grad.size(2), input.size(3) / grad.size(2)})
        .permute({2, 0, 1})
        .reshape({weight.size(3), input.size(0) * input.size(3) / weight.size(3)})
        .permute({1, 0});
    at::Tensor mm_result = at::matmul(mm_input, mm_other);
    grad_weight = mm_result.reshape({grad.size(1), 1, 1, weight.size(3)});
    return grad_weight;
  }

  c10::SmallVector<int64_t, N> dim_list = op_infer::array_to_small_vector(weight.sizes());
  c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";

  at_npu::native::OpCommand cmd;
  cmd.Name("Conv2DBackpropFilter")
      .Input(input, "x", ACL_FORMAT_NCHW)
      .Input(dim_list, at::kInt)
      .Input(grad, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_weight, "y", ACL_FORMAT_NCHW)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_weight;
}

at::Tensor& conv2d_backward_bias_out_nocheck(
    at::Tensor& grad_bias,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  if (grad.numel() == grad.size(0) * grad.size(1)) {
    at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1)});
    op_plugin::sum_out(grad_view, c10::SmallVector<int64_t, N>{0}, false, grad_view.scalar_type(), grad_bias);
  } else {
    at::Tensor grad_view = grad.contiguous().view({grad.size(0), grad.size(1), -1});
    op_plugin::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2}, false, grad_view.scalar_type(), grad_bias);
  }

  return grad_bias;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> conv2d_backward_out_nocheck(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  if (grad_input_mask[0]) {
    conv2d_backward_input_out_nocheck(grad_input, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[1]) {
    conv2d_backward_weight_out_nocheck(grad_weight, input, grad, weight, stride, padding, dilation, groups);
  }

  if (grad_input_mask[2]) {
    conv2d_backward_bias_out_nocheck(grad_bias, input, grad, weight, stride, padding, dilation, groups);
  }

  return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace


std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv2d_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  auto output_sizes = op_infer::conv2d_backward_npu_output_size(input, grad, weight, stride, padding, dilation, groups);

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (grad_input_mask[0]) {
    int64_t grad_input_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
    grad_input = npu_preparation::ApplyTensorWithFormat(
        std::get<0>(output_sizes), input.options(), grad_input_format);
  }

  if (grad_input_mask[1]) {
    // For group conv2d: keep consistent with weight to avoid allreduce accuracy problem.
    // For more info: https://gitee.com/ascend/pytorch-develop/pulls/2255
    if (groups > 1) {
      grad_weight = npu_preparation::ApplyTensorWithFormat(
          std::get<1>(output_sizes),
          weight.options().dtype(at::kFloat),
          ACL_FORMAT_NCHW);
    } else {
      grad_weight = npu_preparation::ApplyTensorWithFormat(
          std::get<1>(output_sizes),
          weight.options().dtype(at::kFloat),
          calcu_op_util::GetTensorNpuFormat(weight));
    }
  }

  if (grad_input_mask[2]) {
    grad_bias = npu_preparation::ApplyTensorWithFormat(
        std::get<2>(output_sizes), grad.options(), ACL_FORMAT_NCHW);
  }

  conv2d_backward_out_nocheck(
      grad_input,
      grad_weight,
      grad_bias,
      input,
      grad,
      weight,
      stride,
      padding,
      dilation,
      groups,
      grad_input_mask);

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
} // namespace op_plugin
