// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void max_pool3d_with_indices_parameter_check(const at::Tensor &self, at::IntArrayRef kernel_size,
                                             at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation)
{
    TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
                "max_pool3d: kernel_size must either be a single int, or a tuple of three ints", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
                "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
                "max_pool3d: padding must be either be a single int, or a tuple of three ints", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
                "max_pool3d: dilation must be either a single int, or a tuple of three ints", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
                "non-empty 4D or 5D (batch mode) tensor expected for input", OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor, at::Tensor> exec_max_pool3d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    max_pool3d_with_indices_parameter_check(self, kernel_size, stride, padding, dilation);

    const int k_D = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_H = kernel_size.size() == 1 ? k_D : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
    const int k_W = kernel_size.size() == 1 ? k_D : at::native::safe_downcast<int, int64_t>(kernel_size[2]);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_D = stride.empty() ? k_D : at::native::safe_downcast<int, int64_t>(stride[0]);
    const int d_H = stride.empty() ? k_H :
                    stride.size() == 1 ? d_D :
                                         at::native::safe_downcast<int, int64_t>(stride[1]);
    const int d_W = stride.empty() ? k_W :
                    stride.size() == 1 ? d_D :
                                         at::native::safe_downcast<int, int64_t>(stride[2]);

    const int pad_D = at::native::safe_downcast<int, int64_t>(padding[0]);
    const int pad_H = padding.size() == 1 ? pad_D : at::native::safe_downcast<int, int64_t>(padding[1]);
    const int pad_W = padding.size() == 1 ? pad_D : at::native::safe_downcast<int, int64_t>(padding[2]);

    const int dilation_D = at::native::safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_H = dilation.size() == 1 ? dilation_D : at::native::safe_downcast<int, int64_t>(dilation[1]);
    const int dilation_W = dilation.size() == 1 ? dilation_D : at::native::safe_downcast<int, int64_t>(dilation[2]);

    const int64_t n_batch = self.ndimension() == 5 ? self.size(-5) : 1;
    const int64_t n_slices = self.size(-4);
    const int64_t input_time = self.size(-3);
    const int64_t input_height = self.size(-2);
    const int64_t input_width = self.size(-1);

    const int64_t output_time =
                  at::native::pooling_output_shape<int64_t>(input_time, k_D, pad_D, d_D, dilation_D, ceil_mode);
    const int64_t output_height =
                  at::native::pooling_output_shape<int64_t>(input_height, k_H, pad_H, d_H, dilation_H, ceil_mode);
    const int64_t output_width =
                  at::native::pooling_output_shape<int64_t>(input_width, k_W, pad_W, d_W, dilation_W, ceil_mode);

    at::native::pool3d_shape_check(self,
                                   n_slices,
                                   k_D, k_H, k_W,
                                   d_D, d_H, d_W,
                                   pad_D, pad_H, pad_W,
                                   dilation_D, dilation_H, dilation_W,
                                   input_time, input_height, input_width,
                                   output_time, output_height, output_width,
                                   "max_pool3d_with_indices");

    c10::SmallVector<int64_t, SIZE> output_size =
         self.ndimension() == 5 ?
                            c10::SmallVector<int64_t, SIZE>({n_batch, n_slices, output_time, output_height, output_width}) :
                            c10::SmallVector<int64_t, SIZE>({n_slices, output_time, output_height, output_width});

    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, self.options());

    // The indices tensor can only be of int32 type
    at::Tensor indices = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnMaxPool3dWithArgmax, self, kernel_size,
                 stride, padding, dilation, ceil_mode, output, indices);

    return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode)
{
    DO_COMPATIBILITY(aclnnMaxPool3dWithArgmax, acl_op::max_pool3d_with_indices(self, kernel_size, stride,
                                                                               padding, dilation, ceil_mode));

    static const bool is_supported = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                      c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                      c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4;
    if (!is_supported) {
        return acl_op::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    }

    return op_api::exec_max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor&, at::Tensor&> max_pool3d_with_indices_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices)
{
    DO_COMPATIBILITY(aclnnMaxPool3dWithArgmax, acl_op::max_pool3d_with_indices_out(self, kernel_size, stride, padding,
                                                                                   dilation, ceil_mode, output, indices));

    static const bool is_supported = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                      c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                      c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4;
    if (!is_supported) {
        return acl_op::max_pool3d_with_indices_out(self, kernel_size, stride, padding, dilation, ceil_mode, output, indices);
    }

    EXEC_NPU_CMD(aclnnMaxPool3dWithArgmax, self, kernel_size, stride, padding, dilation, ceil_mode, output, indices);
    return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}
}
