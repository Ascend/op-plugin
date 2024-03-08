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

namespace {
c10::SmallVector<int64_t, SIZE> conv3d_npu_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                       const at::Tensor &bias, at::IntArrayRef stride,
                                                       at::IntArrayRef padding, at::IntArrayRef dilation,
                                                       int64_t groups)
{
    TORCH_CHECK(input.dim() >= 5, "input has to be more than 5D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 5, "weight has to more than 5D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 3, "dilation has to contain more than 3 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] * stride[2] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));

    int64_t N = input.size(0);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);
    int64_t Do = (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t Ho = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    int64_t Wo = (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1;

    TORCH_CHECK(Do > 0, "Do has to be positive, but got ", Do,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo,
        OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Do, Ho, Wo};
    return output_size;
}

at::Tensor &conv3d_out_nocheck(at::Tensor &result, const at::Tensor &input, const at::Tensor &weight,
                               const at::Tensor &bias, at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 3, "dilation has to contain more than 3 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor filter = weight.to(input.dtype());
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv3D").Input(input, "x").Input(filter, "filter");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(result, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", (string) "NCDHW")
        .Run();

    return result;
}

std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> slow_conv3d_npu_output_size(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, at::IntArrayRef stride,
    at::IntArrayRef padding)
{
    TORCH_CHECK(input.dim() >= 5, "input has to be more than 5D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 5, "weight has to more than 5D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] * stride[2] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t D = input.size(2);
    int64_t H = input.size(3);
    int64_t W = input.size(4);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);
    int64_t Do = (D + 2 * padding[0] - (kernel_size[0])) / stride[0] + 1;
    int64_t Ho = (H + 2 * padding[1] - (kernel_size[1])) / stride[1] + 1;
    int64_t Wo = (W + 2 * padding[2] - (kernel_size[2])) / stride[2] + 1;

    TORCH_CHECK(Do > 0, "Do has to be positive, but got ", Do,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo,
        OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Do, Ho, Wo};
    c10::SmallVector<int64_t, SIZE> finput_size = {N, C * kernel_size[0] * kernel_size[1] * kernel_size[2],
                                                   Do * Ho * Wo};

    return std::tie(output_size, finput_size);
}

at::Tensor &slow_conv3d_forward_npu_nocheck(at::Tensor &result, const at::Tensor &input, const at::Tensor &weight,
                                            at::IntArrayRef kernel_size, const at::Tensor &bias, at::IntArrayRef stride,
                                            at::IntArrayRef padding)
{
    TORCH_CHECK(stride.size() >= 3, "stride has to contain more than 3 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 3, "padding has to contain more than 3 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));

    at::Tensor filter = weight.to(input.dtype());
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1], stride[2]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1, 1};

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv3D").Input(input, "x").Input(filter, "filter");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(result, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("data_format", (string) "NCDHW")
        .Run();

    return result;
}
} // namespace

at::Tensor &npu_conv3d_out(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                           at::Tensor &result)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    auto output_size = conv3d_npu_output_size(input, weight, bias, stride, padding, dilation, groups);
    npu_preparation::CheckOut({input, weight, bias}, result, input, output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        conv3d_out_nocheck(contiguous_result, input, weight, bias, stride, padding, dilation, groups);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        conv3d_out_nocheck(result, input, weight, bias, stride, padding, dilation, groups);
    }
    return result;
}

at::Tensor npu_conv3d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt,
                      at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    auto output_size = conv3d_npu_output_size(input, weight, bias, stride, padding, dilation, groups);
    at::Tensor result = npu_preparation::apply_tensor(input, output_size);
    conv3d_out_nocheck(result, input, weight, bias, stride, padding, dilation, groups);
    return result;
}

at::Tensor &slow_conv3d_forward_out(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                    const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                    at::IntArrayRef padding, at::Tensor &result)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    auto output_size = slow_conv3d_npu_output_size(input, weight, bias, stride, padding);
    npu_preparation::CheckOut({input, weight, bias}, result, input, std::get<0>(output_size));
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        slow_conv3d_forward_npu_nocheck(contiguous_result, input, weight, kernel_size, bias, stride, padding);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        slow_conv3d_forward_npu_nocheck(result, input, weight, kernel_size, bias, stride, padding);
    }
    return result;
}

at::Tensor slow_conv3d_forward(const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
                               const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                               at::IntArrayRef padding)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    auto output_size = slow_conv3d_npu_output_size(self, weight, bias, stride, padding);
    auto result = npu_preparation::apply_tensor_with_format(self, std::get<0>(output_size), ACL_FORMAT_NDC1HWC0);

    slow_conv3d_forward_npu_nocheck(result, self, weight, kernel_size, bias, stride, padding);
    return result;
}

at::Tensor slow_conv3d(const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
                       const c10::optional<at::Tensor> &bias, at::IntArrayRef stride, at::IntArrayRef padding)
{
    return acl_op::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}

at::Tensor &slow_conv3d_out(const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
                            const c10::optional<at::Tensor> &bias, at::IntArrayRef stride, at::IntArrayRef padding,
                            at::Tensor &result)
{
    return acl_op::slow_conv3d_forward_out(self, weight, kernel_size, bias, stride, padding, result);
}
} // namespace acl_op
