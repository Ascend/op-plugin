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
const int MIN_ELEMENT_3D = 3;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor &conv_transpose3d_backward_input_out_nocheck(
    at::Tensor &grad_input, const at::Tensor &grad_output, const at::Tensor &weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= MIN_ELEMENT_3D,
                "stride has to contain more than 3 elements, but got ", stride.size(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= MIN_ELEMENT_3D,
                "padding has to contain more than 3 elements, but got ", padding.size(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= MIN_ELEMENT_3D,
                "dilation has to contain more than 3 elements, but got ", dilation.size(),
                OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
    string data_format = "NCDHW";

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv3D")
        .Input(grad_output, "x")
        .Input(weight, "filter")
        .Output(grad_input, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_format)
        .Run();
    return grad_input;
}

at::Tensor &conv_transpose3d_backward_weight_out_nocheck(
    at::Tensor &grad_weight, const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= MIN_ELEMENT_3D,
                "stride has to contain more than 3 elements, but got ", stride.size(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= MIN_ELEMENT_3D,
                "padding has to contain more than 3 elements, but got ", padding.size(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= MIN_ELEMENT_3D,
                "dilation has to contain more than 3 elements, but got ", dilation.size(),
                OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
    string data_format = "NCDHW";
    at::IntArrayRef input_size = weight.sizes();

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv3DBackpropFilter")
        .Input(grad_output, "x")
        .Input(input_size, at::kInt)
        .Input(input, "out_backprop")
        .Output(grad_weight, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_format)
        .Run();
    return grad_weight;
}

at::Tensor &conv_transpose3d_backward_bias_out_nocheck(at::Tensor &grad_bias, const at::Tensor &grad_output)
{
    TORCH_CHECK(grad_output.dim() >= MIN_ELEMENT_3D,
                "grad_output has to be more than 3D, but got Tensor of dimension ",
                grad_output.dim(), OPS_ERROR(ErrCode::PARAM));
    at::Tensor gradView =
        grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), grad_output.size(2), -1});
    acl_op::sum_out(gradView, c10::SmallVector<int64_t, N>{0, 2, 3}, false, gradView.scalar_type(), grad_bias);
    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> conv_transpose3d_backward_out_nocheck(
    at::Tensor &grad_input, at::Tensor &grad_weight, at::Tensor &grad_bias, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask)
{
    if (output_mask[0]) {
        conv_transpose3d_backward_input_out_nocheck(grad_input, grad_output, weight, padding,
                                                    stride, dilation, groups);
    }
    if (output_mask[1]) {
        conv_transpose3d_backward_weight_out_nocheck(grad_weight, input, grad_output, weight, padding,
                                                     stride, dilation, groups);
    }
    if (output_mask[2]) {
        conv_transpose3d_backward_bias_out_nocheck(grad_bias, grad_output);
    }
    return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward(
    const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask)
{
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    if (output_mask[0]) {
        grad_input = npu_preparation::apply_tensor_with_format(input, ACL_FORMAT_NDC1HWC0);
    }
    if (output_mask[1]) {
        grad_weight = npu_preparation::apply_tensor_with_format(weight.sizes(), weight.options().dtype(at::kFloat),
                                                                npu_preparation::get_tensor_npu_format(weight));
    }
    if (output_mask[2]) {
        grad_bias =
            npu_preparation::apply_tensor_with_format({grad_output.size(1)}, grad_output.options(), ACL_FORMAT_NCDHW);
    }

    conv_transpose3d_backward_out_nocheck(grad_input, grad_weight, grad_bias, input, grad_output, weight,
                                          padding, stride, dilation, groups, output_mask);
    return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace acl_op
