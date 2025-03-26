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
void im2col_shape_check(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                        at::IntArrayRef padding, at::IntArrayRef stride)
{
    bool valid_dims = self.size(1) != 0 && self.size(2) != 0;
    int64_t ndim = self.dim();
    TORCH_CHECK((ndim == 3 && self.size(0) != 0 && valid_dims) || (ndim == 4 && valid_dims && self.size(3) != 0),
                "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for "
                "input, but got: ",
                self.sizes(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
                "im2col: kernel_size must either be a single int, or a tuple of two ints"
                + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
                "im2col: stride must either be omitted, a single int, or a tuple of two ints"
                + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.empty() || dilation.size() == 1 || dilation.size() == 2,
                "im2col: dilation must either be omitted, a single int, or a tuple of two ints"
                + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.empty() || padding.size() == 1 || padding.size() == 2,
                "im2col: padding must either be omitted, a single int, or a tuple of two ints"
                + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor& im2col_out_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef kernel_size,
                               at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride)
{
    if (kernel_size.size() == 1) {
        c10::SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
        kernel_size = at::IntArrayRef(kernel_sizes);
    }
    c10::SmallVector<int64_t, SIZE> default_size = {1};
    c10::SmallVector<int64_t, SIZE> pads_default_size = {0};
    stride = stride.empty() ? at::IntArrayRef(default_size) : stride;
    dilation = dilation.empty() ? at::IntArrayRef(default_size) : dilation;
    auto padding_ = padding.empty() ? at::IntArrayRef(pads_default_size) : padding;
    c10::SmallVector<int64_t, SIZE> pads;
    if (padding_.size() == 1) {
        pads = {padding_[0], padding_[0], padding_[0], padding_[0]};
    } else if (padding_.size() == 2) {
        pads = {padding_[0], padding_[0], padding_[1], padding_[1]};
    }

    auto padding_4d = at::IntArrayRef(pads);

    int64_t stride_h = 1;
    int64_t stride_w = 1;
    if (stride.size() == 1) {
        stride_h = stride[0];
        stride_w = stride[0];
    } else if (stride.size() == 2) {
        stride_h = stride[0];
        stride_w = stride[1];
    }

    int64_t dilation_h = 1;
    int64_t dilation_w = 1;
    if (dilation.size() == 1) {
        dilation_h = dilation[0];
        dilation_w = dilation[0];
    } else if (dilation.size() == 2) {
        dilation_h = dilation[0];
        dilation_w = dilation[1];
    }

    c10::SmallVector<int64_t, N> kernel_sizes = {kernel_size[0], kernel_size[1]};
    c10::SmallVector<int64_t, N> stride_sizes = {stride_h, stride_w};
    c10::SmallVector<int64_t, N> dilations_sizes = {dilation_h, dilation_w};
    c10::SmallVector<int64_t, N> pads_size = {padding_4d[0], padding_4d[1], padding_4d[2], padding_4d[3]};
    string padding_mode = "CALCULATED";
    at::Tensor self_op = (npu_preparation::get_tensor_npu_format(self) == ACL_FORMAT_ND) ?
        at_npu::native::custom_ops::npu_format_cast(self, ACL_FORMAT_NCHW) : self;
    at_npu::native::OpCommand cmd;
    cmd.Name("Im2col")
        .Input(self_op, "x")
        .Output(result)
        .Attr("ksizes", kernel_sizes)
        .Attr("strides", stride_sizes)
        .Attr("dilations", dilations_sizes)
        .Attr("padding_mode", padding_mode)
        .Attr("pads", pads_size)
        .Run();
    return result;
}
} // namespace

at::Tensor& im2col_out(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                       at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor& out)
{
    im2col_shape_check(self, kernel_size, dilation, padding, stride);
    at::Tensor self_cp = self.dim() == 3 ? at::unsqueeze(self, 0) : self;
    auto output_size = op_infer::image_to_col_npu_output_size(self_cp, kernel_size, stride, dilation, padding);

    npu_preparation::CheckOut({self_cp}, out, self_cp, output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        im2col_out_nocheck(contiguous_result, self_cp, kernel_size, dilation, padding, stride);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        im2col_out_nocheck(out, self_cp, kernel_size, dilation, padding, stride);
    }

    if (self.dim() == 3) {
        out = at::squeeze(out, 0);
    }
    return out;
}

at::Tensor im2col(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                  at::IntArrayRef padding, at::IntArrayRef stride)
{
    im2col_shape_check(self, kernel_size, dilation, padding, stride);
    at::Tensor self_cp = self.dim() == 3 ? at::unsqueeze(self, 0) : self;
    auto output_size = op_infer::image_to_col_npu_output_size(self_cp, kernel_size, stride, dilation, padding);
    at::Tensor result = npu_preparation::apply_tensor(self_cp, output_size);
    im2col_out_nocheck(result, self_cp, kernel_size, dilation, padding, stride);
    if (self.dim() == 3) {
        result = at::squeeze(result, 0);
    }
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R0, V2R0)
at::Tensor col2im_backward(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride)
{
    return acl_op::im2col(self, kernel_size, dilation, padding, stride);
}

at::Tensor& col2im_backward_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& result)
{
    return acl_op::im2col_out(self, kernel_size, dilation, padding, stride, result);
}
#endif
} // namespace acl_op
