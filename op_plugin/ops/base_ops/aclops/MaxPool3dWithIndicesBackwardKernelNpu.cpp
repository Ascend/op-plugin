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

#include <ATen/native/Pool.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &max_pool3d_with_indices_backward_out_nocheck(at::Tensor &grad_input, const at::Tensor &grad_output,
                                                         const at::Tensor &self, at::IntArrayRef kernel_size,
                                                         at::IntArrayRef stride, at::IntArrayRef padding,
                                                         at::IntArrayRef dilation, bool ceil_mode,
                                                         const at::Tensor &indices)
{
    int64_t stride_T = 1;
    int64_t stride_H = 1;
    int64_t stride_W = 1;
    if (stride.empty()) {
        stride_T = kernel_size[0];
        stride_H = kernel_size[1];
        stride_W = kernel_size[2];
    } else {
        stride_T = stride[0];
        stride_H = stride[1];
        stride_W = stride[2];
    }

    string padstr = "CALCULATED";
    int64_t ds = self.size(-3);
    int64_t hs = self.size(-2);
    int64_t ws = self.size(-1);
    c10::SmallVector<int64_t, SIZE> padrs(padding);
    if (ceil_mode) {
        padrs[0] += op_plugin::utils::complete_pad(ds, padding[0], kernel_size[0], stride_T);
        padrs[1] += op_plugin::utils::complete_pad(hs, padding[1], kernel_size[1], stride_H);
        padrs[2] += op_plugin::utils::complete_pad(ws, padding[2], kernel_size[2], stride_W);
    }
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, kernel_size[0], kernel_size[1], kernel_size[2]};
    c10::SmallVector<int64_t, SIZE> stride_sizes = {1, 1, stride_T, stride_H, stride_W};
    c10::SmallVector<int64_t, SIZE> padding_sizes = {padding[0], padrs[0], padding[1], padrs[1], padding[2], padrs[2]};

    string data_format = "NCDHW";

    at_npu::native::OpCommand cmd;
    cmd.Name("MaxPool3DGrad")
        .Input(self, "orig_x")
        .Input(indices, "orig_y")
        .Input(grad_output, "grads")
        .Output(grad_input, "y")
        .Attr("ksize", kernel_sizes)
        .Attr("strides", stride_sizes)
        .Attr("padding", padstr)
        .Attr("pads", padding_sizes)
        .Attr("data_format", data_format)
        .Run();

    return grad_input;
}

void max_pool3d_with_indices_backward_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                      at::IntArrayRef stride, at::IntArrayRef padding,
                                                      at::IntArrayRef dilation)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "max_pool3d: kernel_size must either be a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
        "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "max_pool3d: padding must be either be a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
        "max_pool3d: dilation must be either a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 5 || self.ndimension() == 4),
        "maxpool3d expected input to be non-empty 5D(batch mode) or 4D tensor",
        "but input has dim: ", self.ndimension(),
        OPS_ERROR(ErrCode::PARAM));
}
} // namespace

at::Tensor &max_pool3d_with_indices_backward_out(const at::Tensor &grad_output, const at::Tensor &self,
                                                 at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                 at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
                                                 const at::Tensor &indices, at::Tensor &grad_input)
{
    max_pool3d_with_indices_backward_parameter_check(self, kernel_size, stride, padding, dilation);

    at::Tensor self_cp = self;
    if (self.ndimension() == 4) {
        self_cp = self_cp.unsqueeze(0);
    }
    auto output_size = op_infer::input_same_output_size(self_cp);
    npu_preparation::CheckOut({self, indices, grad_output}, grad_input, ACL_FORMAT_NDC1HWC0, self_cp.scalar_type(),
                              output_size);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contig_grad_input = npu_utils::format_contiguous(grad_input);
        max_pool3d_with_indices_backward_out_nocheck(contig_grad_input, grad_output, self, kernel_size, stride, padding,
                                                     dilation, ceil_mode, indices);
        npu_utils::format_fresh_view(grad_input, contig_grad_input);
    } else {
        max_pool3d_with_indices_backward_out_nocheck(grad_input, grad_output, self, kernel_size, stride, padding,
                                                     dilation, ceil_mode, indices);
    }
    return grad_input;
}

at::Tensor max_pool3d_with_indices_backward(const at::Tensor &grad_output, const at::Tensor &self,
                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                            at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
                                            const at::Tensor &indices)
{
    max_pool3d_with_indices_backward_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_T = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_H = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    const int k_W = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[2]);
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_T, k_H, k_W};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    const int d_T = stride.empty() ? k_T : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_H = stride.empty()     ? k_H :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[2]);
    c10::SmallVector<int64_t, SIZE> strides = {d_T, d_H, d_W};
    at::IntArrayRef stridess = at::IntArrayRef(strides);

    const int p_T = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int p_H = padding.size() == 1 ? p_T : at::native::safe_downcast<int, int64_t>(padding[1]);
    const int p_W = padding.size() == 1 ? p_T : at::native::safe_downcast<int, int64_t>(padding[2]);
    c10::SmallVector<int64_t, SIZE> paddings = {p_T, p_H, p_W};
    at::IntArrayRef paddingss = at::IntArrayRef(paddings);

    const int dilation_T = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_H = dilation.size() == 1 ? dilation_T : at::native::safe_downcast<int, int64_t>(dilation[1]);
    const int dilation_W = dilation.size() == 1 ? dilation_T : at::native::safe_downcast<int, int64_t>(dilation[2]);
    c10::SmallVector<int64_t, SIZE> dilations = {dilation_T, dilation_H, dilation_W};
    at::IntArrayRef dilationss = at::IntArrayRef(dilations);

    const int64_t nslices = self.size(-4);
    const int64_t itime = self.size(-3);
    const int64_t iheight = self.size(-2);
    const int64_t iwidth = self.size(-1);
    const int64_t otime = grad_output.size(-3);
    const int64_t oheight = grad_output.size(-2);
    const int64_t owidth = grad_output.size(-1);

    at::native::max_pool3d_backward_shape_check(self, grad_output, indices, nslices, k_T, k_H, k_W, d_T, d_H, d_W, p_T,
                                                p_H, p_W, dilation_T, dilation_H, dilation_W, itime, iheight, iwidth,
                                                otime, oheight, owidth, "max_pool3d_with_indices_backward()");
    at::Tensor self_cp = self;
    at::Tensor grad_output_cp = grad_output.clone();
    at::Tensor indices_cp = indices;
    if (self.ndimension() == 4) {
        self_cp = self_cp.unsqueeze(0);
        grad_output_cp = grad_output_cp.unsqueeze(0);
        indices_cp = indices_cp.unsqueeze(0);
    }
    auto output_size = op_infer::input_same_output_size(self_cp);
    at::Tensor grad_input = npu_preparation::apply_tensor_with_format(
        output_size, self_cp.options().dtype(c10::ScalarType::Float), ACL_FORMAT_NDC1HWC0);

    max_pool3d_with_indices_backward_out_nocheck(grad_input, grad_output_cp, self_cp, kernel_sizess, stridess,
                                                 paddingss, dilationss, ceil_mode, indices_cp);
    grad_input = self.ndimension() == 4 ? grad_input.squeeze(0) : grad_input;
    return grad_input;
}
} // namespace acl_op
