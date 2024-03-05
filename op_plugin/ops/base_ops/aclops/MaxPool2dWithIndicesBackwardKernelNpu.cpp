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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
static const int64_t INDICES_TYPE_CONVERT = 2;
static const int64_t BLOCKSIZE = 16;

using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &max_pool2d_with_indices_backward_out_nocheck(at::Tensor &grad_input, const at::Tensor &grad_output,
                                                         const at::Tensor &self, at::IntArrayRef kernel_size,
                                                         at::IntArrayRef stride, at::IntArrayRef padding,
                                                         at::IntArrayRef dilation, bool ceil_mode,
                                                         const at::Tensor &indices)
{
    at::Tensor self_cp = self;
    at::Tensor grad_output_cp = grad_output;
    at::Tensor indices_cp = indices;
    if (self.dim() == 3) {
        self_cp = self.unsqueeze(0);
        grad_output_cp = grad_output.unsqueeze(0);
        indices_cp = indices.unsqueeze(0);
        grad_input.unsqueeze_(0);
    }
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

    if (indices.dtype() == at::kChar) {
        /* Here to fix the situation when the foward is op_api.
        ** The indices of op_api foward, has continuous storage memory, and dtype is int8, format is NCHW, shape is 4D.
        ** To avoid FE insert transdata node, and damage the storage memory, modifing indices's format to NC1HWC0.
        ** And modify the dtype to short to align with argmax of op parameter.
        */
        at::SmallVector<int64_t, N> shape;
        for (int i = 0; i < indices.dim(); i++) {
            shape.emplace_back((i == 3) ? indices.size(i) / INDICES_TYPE_CONVERT / BLOCKSIZE : indices.size(i));
        }
        shape.emplace_back(BLOCKSIZE);
        
        at::Tensor indices_para = indices.view(at::kShort);
        torch_npu::NPUStorageDesc &desc = torch_npu::NPUBridge::GetNpuStorageImpl(indices_para)->npu_desc_;
        desc.npu_format_ = ACL_FORMAT_NC1HWC0;
        desc.storage_sizes_ = shape;
        desc.data_type_ = at::ScalarType::Short;
        desc.origin_format_ = ACL_FORMAT_NCHW;
        desc.base_sizes_ = indices_para.sizes();
        desc.base_strides_ = indices_para.strides();

        cmd.Name("MaxPoolGradWithArgmaxV1")
            .Input(self, "x")
            .Input(grad_output, "grad")
            .Input(indices_para, "argmax", c10::nullopt, "uint16")
            .Output(grad_input, "y")
            .Attr("ksize", kernel_size_new)
            .Attr("strides", strides_size)
            .Attr("pads", paddings)
            .Attr("dilations", dilations)
            .Attr("ceil_mode", ceil_mode)
            .Run();
    } else {
        cmd.Name("MaxPoolGradWithArgmaxV1")
            .Input(self, "x")
            .Input(grad_output, "grad")
            .Input(indices, "argmax", c10::nullopt, "uint16")
            .Output(grad_input, "y")
            .Attr("ksize", kernel_size_new)
            .Attr("strides", strides_size)
            .Attr("pads", paddings)
            .Attr("dilations", dilations)
            .Attr("ceil_mode", ceil_mode)
            .Run();
    }

    if (self.dim() == 3) {
        grad_input.squeeze_(0);
    }
    return grad_input;
}

void max_pool2d_with_indices_backward_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                                      at::IntArrayRef stride, at::IntArrayRef padding,
                                                      at::IntArrayRef dilation)
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

at::Tensor &max_pool2d_with_indices_backward_out(const at::Tensor &grad_output, const at::Tensor &self,
                                                 at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                 at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
                                                 const at::Tensor &indices, at::Tensor &grad_input)
{
    max_pool2d_with_indices_backward_parameter_check(self, kernel_size, stride, padding, dilation);

    npu_preparation::CheckOut({self, grad_output, indices}, grad_input, self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contig_grad_input = npu_utils::format_contiguous(grad_input);
        max_pool2d_with_indices_backward_out_nocheck(contig_grad_input, grad_output, self, kernel_size, stride, padding,
                                                     dilation, ceil_mode, indices);
        npu_utils::format_fresh_view(grad_input, contig_grad_input);
    } else {
        max_pool2d_with_indices_backward_out_nocheck(grad_input, grad_output, self, kernel_size, stride, padding,
                                                     dilation, ceil_mode, indices);
    }
    return grad_input;
}

at::Tensor max_pool2d_with_indices_backward(const at::Tensor &grad_output_var, const at::Tensor &self,
                                            at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                            at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
                                            const at::Tensor &indices)
{
    max_pool2d_with_indices_backward_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_W = kernel_size.size() == 1 ? k_H : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_H, k_W};
    at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_H = stride.empty() ? k_H : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_W = stride.empty()     ? k_W :
                    stride.size() == 1 ? d_H :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    c10::SmallVector<int64_t, SIZE> strides = {d_H, d_W};
    at::IntArrayRef stridess = at::IntArrayRef(strides);

    const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_W = padding.size() == 1 ? pad_H : at::native::safe_downcast<int, int64_t>(padding[1]);
    c10::SmallVector<int64_t, SIZE> paddings = {pad_H, pad_W};
    at::IntArrayRef padss = at::IntArrayRef(paddings);

    const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_W = dilation.size() == 1 ? dilation_H : at::native::safe_downcast<int, int64_t>(dilation[1]);
    c10::SmallVector<int64_t, SIZE> dilations = {dilation_H, dilation_W};
    at::IntArrayRef dilationss = at::IntArrayRef(dilations);

    at::Tensor grad_input = npu_preparation::apply_tensor(self);

    max_pool2d_with_indices_backward_out_nocheck(grad_input, grad_output_var, self, kernel_sizess, stridess, padss,
                                                 dilationss, ceil_mode, indices);

    return grad_input;
}

} // namespace acl_op
