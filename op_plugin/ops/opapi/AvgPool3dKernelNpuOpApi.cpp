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
const int DIMENSION_1D = 1;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int DIMENSION_5D = 5;
using small_vector = c10::SmallVector<int64_t, op_infer::SIZE>;

at::Tensor &avg_pool3d_out_npu_nocheck_opapi(at::Tensor &result, const at::Tensor &self, at::IntArrayRef kernel,
                                             at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                             bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    int64_t s_divisor_override = 0;
    if (divisor_override.has_value()) {
        s_divisor_override = divisor_override.value();
        TORCH_CHECK(s_divisor_override != 0, "divisor must be not zero", OPS_ERROR(ErrCode::VALUE));
    }

    EXEC_NPU_CMD(aclnnAvgPool3d, self, kernel, stride, padding, ceil_mode, count_include_pad, s_divisor_override,
                 result);

    return result;
}

void avg_pool3d_parameter_check(
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    c10::optional<int64_t> divisor_override)
{
    TORCH_CHECK(kernel_size.size() == DIMENSION_1D || kernel_size.size() == DIMENSION_3D,
                "avg_pool3d: kernel_size must be a single int, or a tuple of three ints" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.empty() || stride.size() == DIMENSION_1D || stride.size() == DIMENSION_3D,
                "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() == DIMENSION_1D || padding.size() == DIMENSION_3D,
                "avg_pool3d: padding must be a single int, or a tuple of three ints" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == DIMENSION_4D || self.ndimension() == DIMENSION_5D),
                "non-empty 4D or 5D (batch mode) tensor expected for input" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
                "divisor must be not zero" + OPS_ERROR(ErrCode::VALUE));
}

small_vector calc_avg_pool3d_output_size(const at::Tensor &self, at::IntArrayRef kernel_size,
                                         at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                         bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    // generalize kernels, strides and paddings to 3D-shape
    TORCH_CHECK(
        !kernel_size.empty(),
        "kernel_size must either be a single int, or a tuple of three ints",
        OPS_ERROR(ErrCode::PARAM));
    const int64_t nslices = self.size(-4);
    const int64_t itime = self.size(-3);
    const int64_t iheight = self.size(-2);
    const int64_t iwidth = self.size(-1);
    const int64_t k_d = kernel_size[0];
    const int64_t k_h = kernel_size.size() == 1 ? k_d : kernel_size[1];
    const int64_t k_w = kernel_size.size() == 1 ? k_d : kernel_size[2];
    c10::SmallVector<int64_t, op_infer::SIZE> kernel_sizes = {k_d, k_h, k_w};
    at::IntArrayRef kernels = at::IntArrayRef(kernel_sizes);

    const int64_t s_d = stride.empty() ? k_d : stride[0];
    const int64_t s_h = stride.empty() ? k_h : stride.size() == 1 ? s_d : stride[1];
    const int64_t s_w = stride.empty() ? k_w : stride.size() == 1 ? s_d : stride[2];
    c10::SmallVector<int64_t, op_infer::SIZE> stride_sizes = {s_d, s_h, s_w};
    TORCH_CHECK(s_d != 0 && s_h != 0 && s_w != 0, "stride should not be zero", OPS_ERROR(ErrCode::VALUE));
    at::IntArrayRef strides = at::IntArrayRef(stride_sizes);

    const int64_t pad_d = padding[0];
    const int64_t pad_h = padding.size() == 1 ? pad_d : padding[1];
    const int64_t pad_w = padding.size() == 1 ? pad_d : padding[2];
    const int64_t otime =
        at::native::pooling_output_shape<int64_t>(itime, k_d, pad_d, s_d, 1, ceil_mode);
    const int64_t oheight =
        at::native::pooling_output_shape<int64_t>(iheight, k_h, pad_h, s_h, 1, ceil_mode);
    const int64_t owidth =
        at::native::pooling_output_shape<int64_t>(iwidth, k_w, pad_w, s_w, 1, ceil_mode);
    at::native::pool3d_shape_check(
        self,
        nslices,
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        pad_d, pad_h, pad_w,
        1, 1, 1,
        itime, iheight, iwidth,
        otime, oheight, owidth,
        "avg_pool3d()",
        true);
    
    c10::SmallVector<int64_t, op_infer::SIZE> padding_sizes = {pad_d, pad_h, pad_w};
    at::IntArrayRef paddings = at::IntArrayRef(padding_sizes);

    auto output_size = op_infer::avg_pool3d_npu_output_size(self, kernels, strides, paddings, ceil_mode);

    return output_size;
}

at::Tensor &avg_pool3d_out(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                           at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                           c10::optional<int64_t> divisor_override, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnAvgPool3d, acl_op::avg_pool3d_out(self, kernel_size, stride, padding, ceil_mode,
                                                            count_include_pad, divisor_override, result));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::avg_pool3d_out(self, kernel_size, stride, padding, ceil_mode,
                                      count_include_pad, divisor_override, result);
    }
    avg_pool3d_parameter_check(self, kernel_size, stride, padding, divisor_override);
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = calc_avg_pool3d_output_size(
        self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    at_npu::native::OpPreparation::check_tensor({self}, result, self, output_size);

    avg_pool3d_out_npu_nocheck_opapi(result, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                     divisor_override);

    return result;
}

at::Tensor avg_pool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override)
{
    DO_COMPATIBILITY(aclnnAvgPool3d, acl_op::avg_pool3d(self, kernel_size, stride, padding, ceil_mode,
                                                        count_include_pad, divisor_override));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    }
    avg_pool3d_parameter_check(self, kernel_size, stride, padding, divisor_override);
    c10::SmallVector<int64_t, op_infer::SIZE> output_size = calc_avg_pool3d_output_size(
        self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);

    avg_pool3d_out_npu_nocheck_opapi(result, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                     divisor_override);

    return result;
}

}
