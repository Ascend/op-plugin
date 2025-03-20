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
void avg_pool3d_backward_out_nocheck(at::Tensor &grad_output, const at::Tensor &grad_input, const at::Tensor &self,
                                     at::IntArrayRef kernel_sizess, at::IntArrayRef stridess, at::IntArrayRef paddingss,
                                     bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    at::Tensor input = self;
    at::Tensor grads = grad_input.contiguous();

    grad_output.resize_as_(input);
    grad_output.zero_();
    if (self.ndimension() == 4) {
        input = input.unsqueeze(0);
        grads = grads.unsqueeze(0);
        grad_output = grad_output.unsqueeze(0);
    }

    TORCH_CHECK(paddingss.size() >= 3, "padding length shoud be at least 3, but got: ", paddingss.size(),
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> dim_list(input.sizes());
    c10::SmallVector<int64_t, N> pads = {paddingss[0], paddingss[0], paddingss[1],
                                         paddingss[1], paddingss[2], paddingss[2]};

    at_npu::native::OpCommand cmd;
    cmd.Name("AvgPool3DGrad")
        .Input(dim_list)
        .Input(grads, "grads")
        .Output(grad_output, "output")
        .Attr("ksize", kernel_sizess)
        .Attr("strides", stridess)
        .Attr("pads", pads)
        .Attr("ceil_mode", ceil_mode)
        .Attr("count_include_pad", count_include_pad);

    if (divisor_override.has_value()) {
        cmd.Attr("divisor_override", divisor_override.value());
    }

    cmd.Attr("data_format", static_cast<string>("NCDHW")).Run();

    if (self.ndimension() == 4) {
        grad_output = grad_output.squeeze(0);
    }
}
void avg_pool3d_backward_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                         at::IntArrayRef padding, c10::optional<int64_t> divisor_override)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "avg_pool3d_backward: kernel_size must be a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
        "avg_pool3d_backward: stride must be omitted, a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
        "avg_pool3d_backward: padding must be a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
        "avg_pool3d_backward divisor must be not zero" + OPS_ERROR(ErrCode::PARAM));
}
} // namespace

at::Tensor &avg_pool3d_backward_out(const at::Tensor &grad_output, const at::Tensor &self, at::IntArrayRef kernel_size,
                                    at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                    bool count_include_pad, c10::optional<int64_t> divisor_override,
                                    at::Tensor &grad_input)
{
    avg_pool3d_backward_parameter_check(self, kernel_size, stride, padding, divisor_override);

    const int k_T = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_H = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    const int k_W = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[2]);
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, k_T, k_H, k_W};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    const int d_T = stride.empty() ? k_T : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_H = stride.empty()     ? k_H :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[2]);
    c10::SmallVector<int64_t, SIZE> strides = {1, 1, d_T, d_H, d_W};
    at::IntArrayRef stridess = at::IntArrayRef(strides);

    const int pad_T = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_H = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[1]);
    const int pad_W = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[2]);
    c10::SmallVector<int64_t, SIZE> paddings = {pad_H, pad_W, pad_T};
    at::IntArrayRef paddingss = at::IntArrayRef(paddings);

    const int64_t nslices = self.size(-4);
    const int64_t itime = self.size(-3);
    const int64_t iheight = self.size(-2);
    const int64_t iwidth = self.size(-1);
    const int64_t otime = grad_output.size(-3);
    const int64_t oheight = grad_output.size(-2);
    const int64_t owidth = grad_output.size(-1);

    /* XXX shape check behavior from TH */
    const int64_t otime_for_shape_check =
        at::native::pooling_output_shape<int64_t>(itime, k_T, pad_T, d_T, 1, ceil_mode);
    const int64_t oheight_for_shape_check =
        at::native::pooling_output_shape<int64_t>(iheight, k_H, pad_H, d_H, 1, ceil_mode);
    const int64_t owidth_for_shape_check =
        at::native::pooling_output_shape<int64_t>(iwidth, k_W, pad_W, d_W, 1, ceil_mode);

    at::native::avg_pool3d_backward_shape_check(
        self, grad_output, nslices, k_T, k_H, k_W, d_T, d_H, d_W, pad_T, pad_H, pad_W, itime, iheight, iwidth,
        otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check, "avg_pool3d_backward_out()");

    npu_preparation::CheckOut({grad_output, self}, grad_input, ACL_FORMAT_NCDHW, self.scalar_type(), self.sizes());
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contig_grad_output = npu_utils::format_contiguous(grad_input);
        avg_pool3d_backward_out_nocheck(contig_grad_output, grad_output, self, kernel_sizess, stridess, paddingss,
                                        ceil_mode, count_include_pad, divisor_override);
        npu_utils::format_fresh_view(grad_input, contig_grad_output);
    } else {
        avg_pool3d_backward_out_nocheck(grad_input, grad_output, self, kernel_sizess, stridess, paddingss, ceil_mode,
                                        count_include_pad, divisor_override);
    }

    return grad_input;
}

at::Tensor avg_pool3d_backward(const at::Tensor &grad_output, const at::Tensor &self, at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override)
{
    at::Tensor input = self;
    at::Tensor grad_input = grad_output;
    if (self.ndimension() == 4) {
        input = self.unsqueeze(0);
        grad_input = grad_input.unsqueeze(0);
    }
    avg_pool3d_backward_parameter_check(input, kernel_size, stride, padding, divisor_override);

    const int k_T = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_H = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    const int k_W = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[2]);
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, k_T, k_H, k_W};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    const int d_T = stride.empty() ? k_T : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_H = stride.empty()     ? k_H :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_T :
                                         at::native::safe_downcast<int, int64_t>(stride[2]);
    c10::SmallVector<int64_t, SIZE> strides = {1, 1, d_T, d_H, d_W};
    at::IntArrayRef stridess = at::IntArrayRef(strides);

    const int pad_T = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_H = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[1]);
    const int pad_W = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[2]);
    c10::SmallVector<int64_t, SIZE> paddings = {pad_H, pad_W, pad_T};
    at::IntArrayRef paddingss = at::IntArrayRef(paddings);

    const int64_t nslices = input.size(-4);
    const int64_t itime = input.size(-3);
    const int64_t iheight = input.size(-2);
    const int64_t iwidth = input.size(-1);
    const int64_t otime = grad_input.size(-3);
    const int64_t oheight = grad_input.size(-2);
    const int64_t owidth = grad_input.size(-1);

    /* XXX shape check behavior from TH */
    const int64_t otime_for_shape_check =
        at::native::pooling_output_shape<int64_t>(itime, k_T, pad_T, d_T, 1, ceil_mode);
    const int64_t oheight_for_shape_check =
        at::native::pooling_output_shape<int64_t>(iheight, k_H, pad_H, d_H, 1, ceil_mode);
    const int64_t owidth_for_shape_check =
        at::native::pooling_output_shape<int64_t>(iwidth, k_W, pad_W, d_W, 1, ceil_mode);

    at::native::avg_pool3d_backward_shape_check(
        input, grad_input, nslices, k_T, k_H, k_W, d_T, d_H, d_W, pad_T, pad_H, pad_W, itime, iheight, iwidth,
        otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check, "avg_pool3d_backward_out()");

    at::Tensor output = npu_preparation::apply_tensor_with_format(input, ACL_FORMAT_NCDHW);
    avg_pool3d_backward_out_nocheck(output, grad_input, input, kernel_sizess, stridess, paddingss, ceil_mode,
                                    count_include_pad, divisor_override);

    if (input.ndimension() == 4) {
        output = output.squeeze(0);
    }

    return output;
}

} // namespace acl_op
