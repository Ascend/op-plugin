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
const int TUPLE_ELEMENTS = 2;
const int DIVIDE_TO_HALF = 2;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int DIVISOR_OVERRIDE_LIMITE = 255;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

c10::SmallVector<int64_t, N> get_paddings(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode)
{
    int64_t pad_down = padding[0];
    int64_t pad_right = padding[1];
    int H = self.size(-2);
    int W = self.size(-1);

    int64_t totalH = H + 2 * padding[0] - kernel_size[0];
    int64_t totalW = W + 2 * padding[1] - kernel_size[1];
    TORCH_CHECK(totalH <= std::numeric_limits<int64_t>::max(), "Large padding causing data overflow"
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(totalW <= std::numeric_limits<int64_t>::max(), "Large padding causing data overflow"
        + OPS_ERROR(ErrCode::VALUE));
    int64_t kH = op_infer::CeilDiv(totalH, stride[0]) + 1;
    int64_t kW = op_infer::CeilDiv(totalW, stride[1]) + 1;
    if (ceil_mode) {
        if ((kH - 1) * stride[0] >= H + padding[0]) {
            --kH;
            int64_t need_pad_h = (kH - 1) * stride[0] + kernel_size[0] - H;
            pad_down = need_pad_h - padding[0];
        }
        if ((kW - 1) * stride[1] >= W + padding[1]) {
            --kW;
            int64_t need_pad_w = (kW - 1) * stride[1] + kernel_size[1] - W;
            pad_right = need_pad_w - padding[1];
        }
    }

    c10::SmallVector<int64_t, N> pads = {padding[0], pad_down, padding[1], pad_right};
    return pads;
}

at::Tensor& avg_pool2d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override)
{
    if (padding.size() == 1) {
        c10::SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
        padding = at::IntArrayRef(paddings);
    }

    int64_t stride_h = 1;
    int64_t stride_w = 1;
    if (!stride.empty()) {
        stride_h = stride[0];
        stride_w = stride[1];
    }
    c10::SmallVector<int64_t, N> kernel_size_new = {1, 1, kernel_size[0], kernel_size[1]};
    c10::SmallVector<int64_t, N> strides_size_new = {1, 1, stride_h, stride_w};
    c10::SmallVector<int64_t, N> pads = get_paddings(self, kernel_size, stride, padding, ceil_mode);
    bool exclusive = !count_include_pad;
    int64_t divisor_override_value = divisor_override.value_or(0);

    at_npu::native::OpCommand cmd;
    cmd.Name("AvgPoolV2")
        .Input(self)
        .Output(result)
        .Attr("ksize", kernel_size_new)
        .Attr("strides", strides_size_new)
        .Attr("padding_mode", static_cast<string>("CALCULATED"))
        .Attr("pads", pads)
        .Attr("data_format", static_cast<string>("NCHW"))
        .Attr("global_pooling", false)
        .Attr("ceil_mode", ceil_mode);
    if (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Char) {
        cmd.Attr("exclusive", true);
    } else {
        cmd.Attr("exclusive", exclusive);
    }
    cmd.Attr("divisor_override", divisor_override_value)
        .Run();
    return result;
}

void avg_pool2d_parameter_check(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    c10::optional<int64_t> divisor_override)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == TUPLE_ELEMENTS,
        "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == TUPLE_ELEMENTS,
        "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 1 || padding.size() == TUPLE_ELEMENTS,
        "avg_pool2d: padding must either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == DIMENSION_3D || self.ndimension() == DIMENSION_4D),
        "non-empty 2D or 3D (batch mode) tensor expected for input"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (!divisor_override.has_value() ||
        (divisor_override.value() > 0 && divisor_override.value() <= DIVISOR_OVERRIDE_LIMITE)),
        "The value of divisor_override = ", divisor_override.value(), " is invaild, only support [1, 255] at present."
        + OPS_ERROR(ErrCode::VALUE));
}
} // namespace

at::Tensor& avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& out)
{
    at::Tensor self_copy = self;
    if (self.dim() == DIMENSION_3D) {
        self_copy = self_copy.unsqueeze(0);
    }
    TORCH_CHECK(!kernel_size.empty(),
        "kernel_size must either be a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM));
    const int64_t k_h = kernel_size[0];
    const int64_t k_w = kernel_size.size() == 1 ? k_h : kernel_size[1];

    c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    const int64_t d_h = stride.empty() ? k_h : stride[0];
    const int64_t d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : stride[1];

    c10::SmallVector<int64_t, SIZE> stride_sizes = {d_h, d_w};
    TORCH_CHECK(d_h != 0 && d_w != 0, "stride should not be zero", OPS_ERROR(ErrCode::VALUE));
    at::IntArrayRef stridess = at::IntArrayRef(stride_sizes);

    const int64_t pad_h = padding[0];
    const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];

    c10::SmallVector<int64_t, SIZE> padding_sizes = {pad_h, pad_w};
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad should not be less than 0", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(pad_h <= k_h / DIVIDE_TO_HALF && pad_w <= k_w / DIVIDE_TO_HALF,
        "pad should be smaller than or equal to half of kernel size", OPS_ERROR(ErrCode::VALUE));
    at::IntArrayRef paddingss = at::IntArrayRef(padding_sizes);

    auto output_sizes = op_infer::avg_pool2d_npu_output_size(
        self_copy, kernel_sizess, stridess, paddingss, ceil_mode);

    npu_preparation::CheckOut(
        {self},
        out,
        self_copy,
        output_sizes);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contig_result = npu_utils::format_contiguous(out);
        avg_pool2d_out_nocheck(
            contig_result, self_copy, kernel_sizess, stridess,
            paddingss, ceil_mode, count_include_pad, divisor_override);
        npu_utils::format_fresh_view(out, contig_result);
    } else {
        avg_pool2d_out_nocheck(
            out, self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
    }

    if (self.dim() == DIMENSION_3D) {
        out = out.squeeze(0);
    }
    return out;
}

at::Tensor avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override)
{
    avg_pool2d_parameter_check(self, kernel_size, stride, padding, divisor_override);

    at::Tensor self_copy = self;
    if (self.dim() == DIMENSION_3D) {
        self_copy = self_copy.unsqueeze(0);
    }
    TORCH_CHECK(!kernel_size.empty(),
        "kernel_size must either be a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM));
    const int64_t k_h = kernel_size[0];
    const int64_t k_w = kernel_size.size() == 1 ? k_h : kernel_size[1];

    c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    const int64_t d_h = stride.empty() ? k_h : stride[0];
    const int64_t d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : stride[1];

    c10::SmallVector<int64_t, SIZE> stride_sizes = {d_h, d_w};
    TORCH_CHECK(d_h != 0 && d_w != 0, "stride should not be zero", OPS_ERROR(ErrCode::VALUE));
    at::IntArrayRef stridess = at::IntArrayRef(stride_sizes);

    const int64_t pad_h = padding[0];
    const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];

    c10::SmallVector<int64_t, SIZE> padding_sizes = {pad_h, pad_w};
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad should not be less than 0", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(pad_h <= k_h / DIVIDE_TO_HALF && pad_w <= k_w / DIVIDE_TO_HALF,
        "pad should be smaller than or equal to half of kernel size", OPS_ERROR(ErrCode::VALUE));
    at::IntArrayRef paddingss = at::IntArrayRef(padding_sizes);

    auto output_sizes = op_infer::avg_pool2d_npu_output_size(
        self_copy, kernel_sizess, stridess, paddingss, ceil_mode);
    at::Tensor result = npu_preparation::apply_tensor(self_copy, output_sizes);

    avg_pool2d_out_nocheck(
        result, self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
    if (self.dim() == DIMENSION_3D) {
        result = result.squeeze(0);
    }
    return result;
}

} // namespace acl_op
