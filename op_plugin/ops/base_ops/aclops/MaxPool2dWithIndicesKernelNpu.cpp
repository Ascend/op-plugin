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
using tensor_list = std::tuple<at::Tensor &, at::Tensor &>;

namespace {
tensor_list max_pool2d_with_indices_out_nocheck(at::Tensor &output, at::Tensor &indices, const at::Tensor &self,
                                                at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)
{
    at::Tensor self_cp = self.dim() == 3 ? self.unsqueeze(0) : self;
    int64_t stride_H = 1;
    int64_t stride_W = 1;
    if (stride.empty()) {
        stride_H = kernel_size[0];
        stride_W = kernel_size[1];
    } else {
        stride_H = stride[0];
        stride_W = stride[1];
    }

    c10::SmallVector<int64_t, N> kernel_size_new = {1, kernel_size[0], kernel_size[1], 1};
    c10::SmallVector<int64_t, N> strides_size = {1, stride_H, stride_W, 1};
    c10::SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
    c10::SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};

    at_npu::native::OpCommand cmd;
    cmd.Name("MaxPoolWithArgmaxV1")
        .Input(self, "x")
        .Output(output, "y")
        .Output(indices, "argmax", c10::nullopt, "uint16")
        .Attr("ksize", kernel_size_new)
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilation", dilations)
        .Attr("ceil_mode", ceil_mode)
        .Run();

    if (self.dim() == 3) {
        output.squeeze_(0);
        indices.squeeze_(0);
    }
    return std::tie(output, indices);
}

void max_pool2d_with_indices_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                             at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        "max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
        "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
        "max_pool2d: padding must be either be a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
        "max_pool2d: dilation must be either a single int, or a tuple of two ints"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input"
        + OPS_ERROR(ErrCode::PARAM));
}
} // namespace

std::tuple<at::Tensor &, at::Tensor &> max_pool2d_with_indices_out(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                                   at::IntArrayRef stride, at::IntArrayRef padding,
                                                                   at::IntArrayRef dilation, bool ceil_mode,
                                                                   at::Tensor &output, at::Tensor &indices)
{
    max_pool2d_with_indices_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_W = kernel_size.size() == 1 ? k_H : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> ksizes = {k_H, k_W};
    at::IntArrayRef kernel_sizes = at::IntArrayRef(ksizes);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_H = stride.empty() ? k_H : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_H :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    c10::SmallVector<int64_t, SIZE> stride_size = {d_H, d_W};
    at::IntArrayRef strides = at::IntArrayRef(stride_size);

    const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_W = padding.size() == 1 ? pad_H : at::native::safe_downcast<int, int64_t>(padding[1]);
    c10::SmallVector<int64_t, SIZE> padding_size = {pad_H, pad_W};
    at::IntArrayRef paddings = at::IntArrayRef(padding_size);

    const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_W = dilation.size() == 1 ? dilation_H : at::native::safe_downcast<int, int64_t>(dilation[1]);
    c10::SmallVector<int64_t, SIZE> dilation_size = {dilation_H, dilation_W};
    at::IntArrayRef dilations = at::IntArrayRef(dilation_size);

    const int64_t n_batch = self.ndimension() == 4 ? self.size(-4) : 1;
    const int64_t n_input_plane = self.size(-3);
    const int64_t input_height = self.size(-2);
    const int64_t input_width = self.size(-1);

    const int64_t output_height =
        at::native::pooling_output_shape<int64_t>(input_height, k_H, pad_H, d_H, dilation_H, ceil_mode);
    const int64_t output_width =
        at::native::pooling_output_shape<int64_t>(input_width, k_W, pad_W, d_W, dilation_W, ceil_mode);

    at::native::pool2d_shape_check(self, k_H, k_W, d_H, d_W, pad_H, pad_W, dilation_H, dilation_W, n_input_plane,
                                   input_height, input_width, output_height, output_width,
                                   self.suggest_memory_format());

    c10::SmallVector<int64_t, SIZE> output_size = {n_batch, n_input_plane, output_height, output_width};

    const int64_t BLOCKSIZE = 16;
    int64_t mask_H = kernel_sizes[0] * kernel_sizes[1];
    int64_t mask_W = (op_infer::CeilDiv(output_height * output_width, BLOCKSIZE) + 1);
    c10::SmallVector<int64_t, SIZE> indices_size = {n_batch, n_input_plane, mask_H, mask_W};

    npu_preparation::CheckOut({self}, output, self, output_size);
    npu_preparation::CheckOut({self}, indices, ACL_FORMAT_NC1HWC0, indices.scalar_type(), indices_size);
    bool output_match = npu_utils::check_match(&output);
    bool indices_match = npu_utils::check_match(&indices);
    if (!(output_match && indices_match)) {
        at::Tensor contiguous_output = output_match ? output : npu_utils::format_contiguous(output);
        at::Tensor contiguous_indices = indices_match ? indices : npu_utils::format_contiguous(indices);
        max_pool2d_with_indices_out_nocheck(contiguous_output, contiguous_indices, self, kernel_sizes,
                                            strides, paddings, dilations, ceil_mode);
        if (!output_match) {
            npu_utils::format_fresh_view(output, contiguous_output);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(indices, contiguous_indices);
        }
    } else {
        max_pool2d_with_indices_out_nocheck(output, indices, self, kernel_sizes, strides, paddings,
                                            dilations, ceil_mode);
    }

    return std::tie(output, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                           at::IntArrayRef stride, at::IntArrayRef padding,
                                                           at::IntArrayRef dilation, bool ceil_mode)
{
    max_pool2d_with_indices_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_W = kernel_size.size() == 1 ? k_H : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> ksizes = {k_H, k_W};
    at::IntArrayRef kernel_sizes = at::IntArrayRef(ksizes);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_H = stride.empty() ? k_H : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_H :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    c10::SmallVector<int64_t, SIZE> stride_size = {d_H, d_W};
    at::IntArrayRef strides = at::IntArrayRef(stride_size);

    const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_W = padding.size() == 1 ? pad_H : at::native::safe_downcast<int, int64_t>(padding[1]);
    c10::SmallVector<int64_t, SIZE> padding_size = {pad_H, pad_W};
    at::IntArrayRef paddings = at::IntArrayRef(padding_size);

    const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_W = dilation.size() == 1 ? dilation_H : at::native::safe_downcast<int, int64_t>(dilation[1]);
    c10::SmallVector<int64_t, SIZE> dilation_size = {dilation_H, dilation_W};
    at::IntArrayRef dilations = at::IntArrayRef(dilation_size);

    const int64_t n_batch = self.ndimension() == 4 ? self.size(-4) : 1;
    const int64_t n_input_plane = self.size(-3);
    const int64_t input_height = self.size(-2);
    const int64_t input_width = self.size(-1);

    const int64_t output_height =
        at::native::pooling_output_shape<int64_t>(input_height, k_H, pad_H, d_H, dilation_H, ceil_mode);
    const int64_t output_width =
        at::native::pooling_output_shape<int64_t>(input_width, k_W, pad_W, d_W, dilation_W, ceil_mode);

    at::native::pool2d_shape_check(self, k_H, k_W, d_H, d_W, pad_H, pad_W, dilation_H, dilation_W, n_input_plane,
                                   input_height, input_width, output_height, output_width,
                                   self.suggest_memory_format());

    c10::SmallVector<int64_t, SIZE> output_size = {n_batch, n_input_plane, output_height, output_width};

    const int64_t BLOCKSIZE = 16;
    int64_t mask_H = kernel_sizes[0] * kernel_sizes[1];
    int64_t mask_W = (op_infer::CeilDiv(output_height * output_width, BLOCKSIZE) + 1);
    c10::SmallVector<int64_t, SIZE> indices_size = {n_batch, n_input_plane, mask_H, mask_W};

    at::Tensor output = npu_preparation::apply_tensor(self, output_size);
    at::Tensor indices = npu_preparation::apply_tensor_with_format(self, indices_size, ACL_FORMAT_NC1HWC0, true);

    max_pool2d_with_indices_out_nocheck(output, indices, self, kernel_sizes, strides, paddings,
                                        dilations, ceil_mode);
    return std::make_tuple(output, indices);
}

} // namespace acl_op
