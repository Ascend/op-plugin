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
at::Tensor &conv_transpose2d_backward_input_out_nocheck(at::Tensor &grad_input, const at::Tensor &input,
                                                        const at::Tensor &grad_output, const at::Tensor &weight,
                                                        at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                        at::IntArrayRef stride, at::IntArrayRef dilation,
                                                        int64_t groups)
{
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    // constructs the input and output NPUTensorDesc
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_format = "NCHW";

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2D")
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

at::Tensor &conv_transpose2d_backward_weight_out_nocheck(at::Tensor &grad_weight, const at::Tensor &input,
                                                         const at::Tensor &grad_output, const at::Tensor &weight,
                                                         at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                         at::IntArrayRef stride, at::IntArrayRef dilation,
                                                         int64_t groups)
{
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> dimList = op_infer::array_to_small_vector(weight.sizes());
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_format = "NCHW";

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2DBackpropFilter")
        .Input(grad_output, "x")
        .Input(dimList, at::kInt)
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

at::Tensor &conv_transpose2d_backward_bias_out_nocheck(at::Tensor &grad_bias, const at::Tensor &input,
                                                       const at::Tensor &grad_output, const at::Tensor &weight,
                                                       at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                       at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(grad_output.dim() >= 2, "grad_output has to be more than 2D, but got Tensor of dimension ",
        grad_output.dim(), OPS_ERROR(ErrCode::PARAM));
    at::Tensor grad_view = grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1});
    acl_op::sum_out(grad_view, c10::SmallVector<int64_t, N>{0, 2}, false, grad_view.scalar_type(), grad_bias);
    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> conv_transpose2d_backward_out_nocheck(
    at::Tensor &grad_input, at::Tensor &grad_weight, at::Tensor &grad_bias, const at::Tensor &input,
    const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask)
{
    if (output_mask[0]) {
        conv_transpose2d_backward_input_out_nocheck(grad_input, input, grad_output, weight, padding, output_padding,
                                                    stride, dilation, groups);
    }
    if (output_mask[1]) {
        conv_transpose2d_backward_weight_out_nocheck(grad_weight, input, grad_output, weight, padding, output_padding,
                                                     stride, dilation, groups);
    }
    if (output_mask[2]) {
        conv_transpose2d_backward_bias_out_nocheck(grad_bias, input, grad_output, weight, padding, output_padding,
                                                   stride, dilation, groups);
    }

    return std::tie(grad_input, grad_weight, grad_bias);
}

c10::SmallVector<int64_t, SIZE> convolution_transpose3d_npu_output_size(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(input.dim() >= 5, "input has to be more than 5D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 5, "weight has to be more than 5D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 3, "dilation has to contain more than 3 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    int64_t N = input.size(0);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);
    int64_t Co = weight.size(1) * groups;
    auto kernel_size = weight.sizes().slice(2);

    int64_t Do = (D - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
    int64_t Ho = (H - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;
    int64_t Wo = (W - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_size[2] - 1) + output_padding[2] + 1;

    TORCH_CHECK(Do > 0, "Do has to be positive, but got ", Do, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo, OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Do, Ho, Wo};
    return output_size;
}

at::Tensor &convolution_transpose3d_out_npu_nocheck(at::Tensor &result, const at::Tensor &input,
                                                    const at::Tensor &weight, const at::Tensor &bias,
                                                    at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 3, "dilation has to contain more than 3 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
    c10::SmallVector<int64_t, N> outputpadding = {0, 0, 0, 0, 0};
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
    string data_format = "NCDHW";

    c10::SmallVector<int64_t, N> sizeVec = op_infer::array_to_small_vector(result.sizes());
    at_npu::native::OpCommand cmd;
    cmd.Name("Conv3DTranspose").Input(sizeVec, at::kInt).Input(input).Input(weight);
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(result)
        .Attr("pads", paddings)
        .Attr("output_padding", outputpadding)
        .Attr("strides", strides_size)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_format)
        .Run();

    return result;
}

at::Tensor convolution_transpose3d_nocheck(const at::Tensor &input, const at::Tensor &weight,
                                           const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef padding,
                                           at::IntArrayRef output_padding, at::IntArrayRef stride,
                                           at::IntArrayRef dilation, int64_t groups)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    auto output_size =
        convolution_transpose3d_npu_output_size(input, weight, bias, padding, output_padding, stride, dilation, groups);
    at::Tensor result = npu_preparation::apply_tensor_with_format(input, output_size, ACL_FORMAT_NDC1HWC0);

    convolution_transpose3d_out_npu_nocheck(result, input, weight, bias, padding, output_padding, stride, dilation,
                                            groups);
    return result;
}

at::Tensor convolution_transpose_kernel_nocheck(const at::Tensor &input, const at::Tensor &weight,
                                                const c10::optional<at::Tensor> &bias, at::IntArrayRef padding,
                                                at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                at::IntArrayRef dilation, int64_t groups)
{
    int64_t dim = input.ndimension();
    TORCH_CHECK(dim != 3, " Currently the private format does not support 3D input,"
        " you can try torch.npu.config.allow_internal_format = False to resolve this functional bug"
        + OPS_ERROR(ErrCode::NOT_SUPPORT));
    at::Tensor output;
    if (dim == 4) {
        output = acl_op::npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups);
    } else if (dim == 5) {
        output =
            convolution_transpose3d_nocheck(input, weight, bias, padding, output_padding, stride, dilation, groups);
    }
    return output;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose2d_backward(
    const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &weight, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask)
{
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    if (output_mask[0]) {
        int64_t grad_input_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
        grad_input = npu_preparation::apply_tensor_with_format(input, grad_input_format);
    }
    if (output_mask[1]) {
        grad_weight = npu_preparation::apply_tensor_with_format(weight.sizes(), weight.options().dtype(at::kFloat),
                                                                npu_preparation::get_tensor_npu_format(weight));
    }
    if (output_mask[2]) {
        grad_bias =
            npu_preparation::apply_tensor_with_format({grad_output.size(1)}, grad_output.options(), ACL_FORMAT_NCHW);
    }

    conv_transpose2d_backward_out_nocheck(grad_input, grad_weight, grad_bias, input, grad_output, weight, padding,
                                          output_padding, stride, dilation, groups, output_mask);
    return std::tie(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_transpose_backward(
    const at::Tensor &input, const at::Tensor &grad, const at::Tensor &weight, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> grad_input_mask)
{
    int64_t dim = input.ndimension();
    std::tuple<at::Tensor, at::Tensor, at::Tensor> output;
    if (dim == 4) {
        output = acl_op::npu_conv_transpose2d_backward(input, grad, weight, padding, output_padding, stride, dilation,
                                                       groups, grad_input_mask);
    } else if (dim == 5) {
        output = acl_op::npu_conv_transpose3d_backward(input, grad, weight, padding, output_padding, stride, dilation,
                                                       groups, grad_input_mask);
    }
    // Note:weight.grad should be equal weight
    if (std::get<1>(output).defined()) {
        std::get<1>(output) = at_npu::native::custom_ops::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
    }
    return output;
}

at::Tensor npu_convolution_transpose(const at::Tensor &input, const at::Tensor &weight,
                                     const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef padding,
                                     at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation,
                                     int64_t groups)
{
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value()) {
        if (bias_opt.value().defined()) {
            bias = bias_opt;
        }
    }

    return convolution_transpose_kernel_nocheck(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
} // namespace acl_op
