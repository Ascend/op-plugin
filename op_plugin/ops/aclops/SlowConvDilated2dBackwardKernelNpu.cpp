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

#include <algorithm>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
inline bool all_positive(at::IntArrayRef &arr)
{
    return std::all_of(arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

at::Tensor &slow_conv_dilated2d_backward_input_out_nocheck(at::Tensor &grad_input, const at::Tensor &grad_output,
                                                           const at::Tensor &self, const at::Tensor &weight,
                                                           at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                           at::IntArrayRef padding, at::IntArrayRef dilation)
{
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_formats = "NCHW";
    int64_t groups = 1;
    c10::SmallVector<int64_t, N> dim_list = op_infer::array_to_small_vector(self.sizes());
    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2DBackpropInput")
        .Input(dim_list, at::kInt)
        .Input(weight, "filter")
        .Input(grad_output, "out_backprop")
        .Output(grad_input, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_formats)
        .Run();

    return grad_input;
}

at::Tensor &slow_conv_dilated2d_backward_weight_out_nocheck(at::Tensor &grad_weight, const at::Tensor &grad_output,
                                                            const at::Tensor &self, const at::Tensor &weight,
                                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                            at::IntArrayRef padding, at::IntArrayRef dilation)
{
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    string data_formats = "NCHW";
    int64_t groups = 1;
    c10::SmallVector<int64_t, N> dim_list = op_infer::array_to_small_vector(weight.sizes());

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2DBackpropFilter")
        .Input(self, "x")
        .Input(dim_list, at::kInt)
        .Input(grad_output, "out_backprop")
        .Output(grad_weight)
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_formats)
        .Run();

    return grad_weight;
}

at::Tensor &slow_conv_dilated2d_backward_bias_out_check(at::Tensor &grad_bias, const at::Tensor &grad_output,
                                                        const at::Tensor &self, const at::Tensor &weight,
                                                        at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                        at::IntArrayRef padding, at::IntArrayRef dilation)
{
    string data_formats = "NCHW";
    at_npu::native::OpCommand cmd;
    cmd.Name("BiasAddGrad").Input(self).Output(grad_bias).Attr("data_format", data_formats).Run();

    return grad_bias;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> slow_conv_dilated2d_backward_out_nocheck(
    at::Tensor &grad_input, at::Tensor &grad_weight, at::Tensor &grad_bias, const at::Tensor &grad_output,
    const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool, 3> output_mask)
{
    TORCH_CHECK(kernel_size.size() == 2, "kernel sizes length should be 2, but got ", kernel_size.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 2, "strides length should be 2, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() == 2, "dilations length should be 2, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 2, "pads length should be 2, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(all_positive(kernel_size), "kernel size should be greater than zero, but got ", kernel_size,
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(all_positive(stride), "stride should be greater than zero, but got ", stride,
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(all_positive(dilation), "dilation should be greater than zero, but got ", dilation,
        OPS_ERROR(ErrCode::PARAM));
    if (output_mask[0]) {
        slow_conv_dilated2d_backward_input_out_nocheck(grad_input, grad_output, self, weight, kernel_size, stride,
                                                       padding, dilation);
    }
    if (output_mask[1]) {
        slow_conv_dilated2d_backward_weight_out_nocheck(grad_weight, grad_output, self, weight, kernel_size, stride,
                                                        padding, dilation);
    }
    if (output_mask[2]) {
        slow_conv_dilated2d_backward_bias_out_check(grad_bias, grad_output, self, weight, kernel_size, stride, padding,
                                                    dilation);
    }

    return std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>(grad_input, grad_weight, grad_bias);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> slow_conv_dilated2d_backward(
    const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, std::array<bool, 3> output_mask)
{
    at::Tensor undefined;
    at::Tensor grad_input = (output_mask[0] ? npu_preparation::apply_tensor(grad_output, self.sizes()) : undefined);
    at::Tensor grad_weight = (output_mask[1] ? npu_preparation::apply_tensor(grad_output, weight.sizes()) : undefined);
    at::Tensor grad_bias = (output_mask[2] ? npu_preparation::apply_tensor(grad_output, weight.size(0)) : undefined);

    slow_conv_dilated2d_backward_out_nocheck(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size,
                                             stride, padding, dilation, output_mask);

    return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace acl_op
