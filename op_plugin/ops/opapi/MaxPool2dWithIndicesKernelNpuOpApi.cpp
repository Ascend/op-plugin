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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void max_pool2d_with_indices_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                             at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
                "max_pool2d: kernel_size must either be a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
                "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
                "max_pool2d: padding must be either be a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
                "max_pool2d: dilation must be either a single int, or a tuple of two ints", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
                "non-empty 3D or 4D (batch mode) tensor expected for input", OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor, at::Tensor> exec_max_pool2d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    max_pool2d_with_indices_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_W = kernel_size.size() == 1 ? k_H : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> ksizes = {k_H, k_W};
    at::IntArrayRef kernel_sizes = at::IntArrayRef(ksizes);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_H = stride.empty() ? k_H : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_W = stride.empty() ? k_W :
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

    c10::SmallVector<int64_t, SIZE> output_size =
         self.ndimension() == 4 ?
                            c10::SmallVector<int64_t, SIZE>({n_batch, n_input_plane, output_height, output_width}) :
                            c10::SmallVector<int64_t, SIZE>({n_input_plane, output_height, output_width});

    const int64_t BLOCKSIZE = 16;
    int64_t mask_H = kernel_sizes[0] * kernel_sizes[1];
    int64_t mask_W = (op_infer::CeilDiv(output_height * output_width, BLOCKSIZE) + 1);
    // 5HD format return the indices with mask, dtype int8,size should be muls 16*2
    c10::SmallVector<int64_t, SIZE> indices_size =
         self.ndimension() == 4 ?
                c10::SmallVector<int64_t, SIZE>({n_batch, n_input_plane, mask_H, mask_W * 32}) :
                c10::SmallVector<int64_t, SIZE>({n_input_plane, mask_H, mask_W * 32});

    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options());
    at::Tensor indices = npu_preparation::apply_tensor_without_format(indices_size, self.options().dtype(at::kChar));

    EXEC_NPU_CMD(aclnnMaxPool2dWithMask, self, kernel_sizes,
                 strides, paddings, dilations, ceil_mode, output, indices);

    return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    DO_COMPATIBILITY(aclnnMaxPool2dWithMask,
                     acl_op::max_pool2d_with_indices(self, kernel_size,
                                                     stride, padding, dilation, ceil_mode));

    return op_api::exec_max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor&, at::Tensor&> max_pool2d_with_indices_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices)
{
    DO_COMPATIBILITY(aclnnMaxPool2dWithMask,
                     acl_op::max_pool2d_with_indices_out(self, kernel_size, stride,
                                                         padding, dilation, ceil_mode, output, indices));
    // execute 5HD format
    EXEC_NPU_CMD(aclnnMaxPool2dWithMask, self, kernel_size,
                 stride, padding, dilation, ceil_mode, output, indices);
    return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}
}
