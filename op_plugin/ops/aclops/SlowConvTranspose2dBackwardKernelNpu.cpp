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
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
at::Tensor &slow_conv_transpose2d_backward_grad_output_out_nocheck(
    at::Tensor &grad_input, const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &weight,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef dilation)
{
    TORCH_CHECK(stride.size() >= 2, "stride size must bigger than 2." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding size must bigger than 2." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation size must bigger than 2." + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_formats = "NCHW";
    int64_t groups = 1;
    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2D")
        .Input(grad_output, "x")
        .Input(weight, "filter")
        .Output(grad_input, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_formats)
        .Run();

    return grad_input;
}

at::Tensor &slow_conv_transpose2d_backward_weight_out_nocheck(at::Tensor &grad_weight, const at::Tensor &grad_output,
                                                              const at::Tensor &self, const at::Tensor &weight,
                                                              at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                              at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                              at::IntArrayRef dilation)
{
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_formats = "NCHW";
    int64_t groups = 1;
    c10::SmallVector<int64_t, N> dim_list = op_infer::array_to_small_vector(weight.sizes());
    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2DBackpropFilter")
        .Input(grad_output, "x")
        .Input(dim_list, at::kInt)
        .Input(self, "out_backprop")
        .Output(grad_weight, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_formats)
        .Run();

    return grad_weight;
}

at::Tensor &slow_conv_transpose2d_backward_bias_out_nocheck(at::Tensor &grad_bias, const at::Tensor &grad_output,
                                                            const at::Tensor &self, const at::Tensor &weight,
                                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                            at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                            at::IntArrayRef dilation)
{
    at::Tensor grad_view = grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1});
    acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2}, false, grad_view.scalar_type(), grad_bias);

    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> slow_conv_transpose2d_backward_out_nocheck(
    const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation,
    at::Tensor &grad_input, at::Tensor &grad_weight, at::Tensor &grad_bias)
{
    TORCH_CHECK(dilation.size() >= 2,
        "slow_conv_transpose2d_backward expected dilation greater than or equal to 2D,"
        " but input dilation has sizes ",
        dilation.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2,
        "slow_conv_transpose2d_backward expected dilation greater than or equal to 2D,"
        " but input padding has sizes ",
        padding.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 2,
        "slow_conv_transpose2d_backward expected dilation greater than or equal to 2D,"
        " but input stride has sizes ",
        stride.size(), OPS_ERROR(ErrCode::PARAM));
    slow_conv_transpose2d_backward_grad_output_out_nocheck(grad_input, grad_output, self, weight, kernel_size, stride,
                                                           padding, output_padding, dilation);
    slow_conv_transpose2d_backward_weight_out_nocheck(grad_weight, grad_output, self, weight, kernel_size, stride,
                                                      padding, output_padding, dilation);
    slow_conv_transpose2d_backward_bias_out_nocheck(grad_bias, grad_output, self, weight, kernel_size, stride, padding,
                                                    output_padding, dilation);

    return std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>(grad_input, grad_weight, grad_bias);
}
} // namespace

tensor_list slow_conv_transpose2d_backward(const at::Tensor &grad_output, const at::Tensor &self,
                                           const at::Tensor &weight, at::IntArrayRef kernel_size,
                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                           at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                           std::array<bool, 3> output_mask)
{
    auto flag = 2;
    auto output_sizes = op_infer::slow_conv_transpose2d_backward_npu_output_size(
        grad_output, self, weight);
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    int64_t grad_format = self.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
    if (output_mask[0]) {
        grad_input = npu_preparation::apply_tensor_with_format(self, std::get<0>(output_sizes), grad_format);
    }
    if (output_mask[1]) {
        grad_weight =
            npu_preparation::apply_tensor_with_format(std::get<1>(output_sizes), weight.options().dtype(at::kFloat),
                                                      npu_preparation::get_tensor_npu_format(weight));
    }
    if (output_mask[flag]) {
        grad_bias = npu_preparation::apply_tensor_with_format(grad_output, {grad_output.size(1)}, ACL_FORMAT_NCHW);
    }

    return slow_conv_transpose2d_backward_out_nocheck(grad_output, self, weight, kernel_size, stride, padding,
                                                      output_padding, dilation, grad_input, grad_weight, grad_bias);
}
} // namespace acl_op
