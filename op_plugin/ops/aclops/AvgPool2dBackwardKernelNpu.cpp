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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& avg_pool2d_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad)
{
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    if (!stride.empty()) {
        stride_h = stride[0];
        stride_w = stride[1];
    }
    c10::SmallVector<int64_t, N> kernel_size_new = {1, 1, kernel_size[0], kernel_size[1]};
    c10::SmallVector<int64_t, N> strides_size_new = {1, 1, stride_h, stride_w};
    string padding_mode = "CALCULATED";
    c10::SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};
    string format = "NCHW";
    bool pooling = false;
    bool exclusive = !count_include_pad;

    at_npu::native::OpCommand cmd;
    cmd.Name("AvgPoolV2Grad")
        .Input(self.sizes())
        .Input(grad_output)
        .Output(grad_input)
        .Attr("ksize", kernel_size_new)
        .Attr("strides", strides_size_new)
        .Attr("padding_mode", padding_mode)
        .Attr("pads", pads)
        .Attr("data_format", format)
        .Attr("global_pooling", pooling)
        .Attr("ceil_mode", ceil_mode)
        .Attr("exclusive", exclusive)
        .Run();
    return grad_input;
}
}

at::Tensor& avg_pool2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& grad_input)
{
    // check kernel_size
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    if (kernel_size.size() == 1) {
        c10::SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
        kernel_size = at::IntArrayRef(kernel_sizes);
    }
    // cbeck stride
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
        "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    stride = stride.empty() ? kernel_size : stride;
    // check padding
    TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "avg_pool2d: padding must either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    if (padding.size() == 1) {
        c10::SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
        padding = at::IntArrayRef(paddings);
    }
    // check divisor_override
    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero"
        + OPS_ERROR(ErrCode::VALUE));
    // check the dimensions of the input tensor
    const int64_t ndim = self.ndimension();
    TORCH_CHECK((ndim == 3 || ndim == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input" + OPS_ERROR(ErrCode::PARAM));
    npu_preparation::CheckOut(
        {self, grad_output},
        grad_input,
        self);
    at::Tensor self_4d = (ndim == 3 ? self.unsqueeze(0) : self);
    at::Tensor grad_output_4d = (ndim == 3 ? grad_output.unsqueeze(0) : grad_output);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contig_tensor = npu_utils::format_contiguous(grad_input);
        avg_pool2d_backward_out_npu_nocheck(
            contig_tensor,
            grad_output_4d,
            self_4d,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad);
        npu_utils::format_fresh_view(grad_input, contig_tensor);
    } else {
        avg_pool2d_backward_out_npu_nocheck(
            grad_input,
            grad_output_4d,
            self_4d,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad);
    }
    at::Tensor grad_input_origin_dims = (ndim == 3 ? grad_input.squeeze(0) : grad_input);
    return grad_input_origin_dims;
}

at::Tensor avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);

    acl_op::avg_pool2d_backward_out(
        grad_output,
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        grad_input);
    return grad_input;
}

} // namespace acl_op
