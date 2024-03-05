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
std::tuple<at::Tensor &, at::Tensor &> max_pool3d_with_indices_out_nocheck(at::Tensor &result, const at::Tensor &self,
                                                                           at::IntArrayRef kernel_size,
                                                                           at::IntArrayRef stride, at::IntArrayRef pads,
                                                                           at::IntArrayRef dilation, bool ceil_mode,
                                                                           at::Tensor &indice)
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

    string padding = "CALCULATED";
    int64_t ds = self.size(-3);
    int64_t hs = self.size(-2);
    int64_t ws = self.size(-1);
    c10::SmallVector<int64_t, SIZE> padrs(pads);
    if (ceil_mode) {
        padrs[0] += op_plugin::utils::complete_pad(ds, pads[0], kernel_size[0], stride_T);
        padrs[1] += op_plugin::utils::complete_pad(hs, pads[1], kernel_size[1], stride_H);
        padrs[2] += op_plugin::utils::complete_pad(ws, pads[2], kernel_size[2], stride_W);
    }
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, kernel_size[0], kernel_size[1], kernel_size[2]};
    c10::SmallVector<int64_t, SIZE> stride_sizes = {1, 1, stride_T, stride_H, stride_W};
    c10::SmallVector<int64_t, SIZE> pads_sizes = {pads[0], padrs[0], pads[1], padrs[1], pads[2], padrs[2]};
    c10::SmallVector<int64_t, SIZE> dilation_sizes = {1, 1, dilation[0], dilation[1], dilation[2]};
    string data_format = "NCDHW";

    at_npu::native::OpCommand cmd;
    cmd.Name("MaxPool3D")
        .Input(self)
        .Output(result)
        .Attr("ksize", kernel_sizes)
        .Attr("strides", stride_sizes)
        .Attr("padding", padding)
        .Attr("pads", pads_sizes)
        .Attr("dilation", dilation_sizes)
        .Attr("ceil_mode", (int64_t)ceil_mode)
        .Attr("data_format", data_format)
        .Run();
    return std::tie(result, result);
}

void max_pool3d_with_indices_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                             at::IntArrayRef stride, at::IntArrayRef pads, at::IntArrayRef dilation)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
        "max_pool3d: kernel_size must either be a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
        "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(pads.size() == 1 || pads.size() == 3,
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

c10::SmallVector<int64_t, SIZE> max_pool3d_with_indices_output_size(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                                    at::IntArrayRef stride, at::IntArrayRef pads,
                                                                    at::IntArrayRef dilation, bool ceil_mode)
{
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

    const int p_T = at::native::safe_downcast<int, int64_t>(pads[0]);
    const int p_H = pads.size() == 1 ? p_T : at::native::safe_downcast<int, int64_t>(pads[1]);
    const int p_W = pads.size() == 1 ? p_T : at::native::safe_downcast<int, int64_t>(pads[2]);
    c10::SmallVector<int64_t, SIZE> paddings = {p_T, p_H, p_W};
    at::IntArrayRef padss = at::IntArrayRef(paddings);

    const int dilation_T = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_H = dilation.size() == 1 ? dilation_T : at::native::safe_downcast<int, int64_t>(dilation[1]);
    const int dilation_W = dilation.size() == 1 ? dilation_T : at::native::safe_downcast<int, int64_t>(dilation[2]);
    c10::SmallVector<int64_t, SIZE> dilations = {dilation_T, dilation_H, dilation_W};
    at::IntArrayRef dilationss = at::IntArrayRef(dilations);

    const int64_t nslices = self.size(-4);
    const int64_t itime = self.size(-3);
    const int64_t iheight = self.size(-2);
    const int64_t iwidth = self.size(-1);

    const int64_t otime = at::native::pooling_output_shape<int64_t>(itime, k_T, p_T, d_T, dilation_T, ceil_mode);
    const int64_t oheight = at::native::pooling_output_shape<int64_t>(iheight, k_H, p_H, d_H, dilation_H, ceil_mode);
    const int64_t owidth = at::native::pooling_output_shape<int64_t>(iwidth, k_W, p_W, d_W, dilation_W, ceil_mode);

    at::native::pool3d_shape_check(self, nslices, k_T, k_H, k_W, d_T, d_H, d_W, p_T, p_H, p_W, dilation_T, dilation_H,
                                   dilation_W, itime, iheight, iwidth, otime, oheight, owidth,
                                   "max_pool3d_with_indices()");
    at::Tensor self_cp = self.ndimension() == 4 ? self.unsqueeze(0) : self;
    c10::SmallVector<int64_t, SIZE> output_size = {self_cp.size(0), self_cp.size(1), otime, oheight, owidth};
    return output_size;
}
} // namespace

std::tuple<at::Tensor &, at::Tensor &> max_pool3d_with_indices_out(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                                   at::IntArrayRef stride, at::IntArrayRef pads,
                                                                   at::IntArrayRef dilation, bool ceil_mode,
                                                                   at::Tensor &result, at::Tensor &indice)
{
    max_pool3d_with_indices_parameter_check(self, kernel_size, stride, pads, dilation);

    c10::SmallVector<int64_t, SIZE> output_size =
        max_pool3d_with_indices_output_size(self, kernel_size, stride, pads, dilation, ceil_mode);
    npu_preparation::CheckOut({self}, result, ACL_FORMAT_NDC1HWC0, self.scalar_type(), output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contig_result = npu_utils::format_contiguous(result);
        max_pool3d_with_indices_out_nocheck(contig_result, self, kernel_size, stride, pads, dilation, ceil_mode,
                                            indice);
        npu_utils::format_fresh_view(result, contig_result);
    } else {
        max_pool3d_with_indices_out_nocheck(result, self, kernel_size, stride, pads, dilation, ceil_mode, indice);
    }
    return std::tie(result, result);
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                           at::IntArrayRef stride, at::IntArrayRef pads,
                                                           at::IntArrayRef dilation, bool ceil_mode)
{
    max_pool3d_with_indices_parameter_check(self, kernel_size, stride, pads, dilation);
    at::Tensor self_cp = self.ndimension() == 4 ? self.unsqueeze(0) : self;
    c10::SmallVector<int64_t, SIZE> output_size =
        max_pool3d_with_indices_output_size(self, kernel_size, stride, pads, dilation, ceil_mode);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_NDC1HWC0);

    max_pool3d_with_indices_out_nocheck(result, self, kernel_size, stride, pads, dilation, ceil_mode, result);
    result = self.ndimension() == 4 ? result.squeeze(0) : result;
    return std::tie(result, result);
}

} // namespace acl_op
