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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& upsample_bilinear2d_out(const at::Tensor& self_ex, at::IntArrayRef output_size,
                                    bool align_corners, c10::optional<double> scales_h,
                                    c10::optional<double> scales_w, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2d, acl_op::upsample_bilinear2d_out(self_ex, output_size, align_corners,
                                                                              scales_h, scales_w, result));
    at::Tensor self = self_ex;
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Double, "upsample_binlinear_2d not support torch.fp64 dtypes",
                OPS_ERROR(ErrCode::TYPE));
    auto outputSize = op_infer::upsample_bilinear2d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, result, self, outputSize);
    double scales_h_attr = scales_h.value_or(1);
    double scales_w_attr = scales_w.value_or(1);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2d, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

at::Tensor upsample_bilinear2d(const at::Tensor& self_ex, at::IntArrayRef output_size,
                               bool align_corners, c10::optional<double> scales_h,
                               c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2d,
                     acl_op::upsample_bilinear2d(self_ex, output_size, align_corners, scales_h, scales_w));
    at::Tensor self = self_ex;
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Double, "upsample_binlinear_2d not support torch.fp64 dtypes",
                OPS_ERROR(ErrCode::TYPE));
    double scales_h_attr = scales_h.value_or(1);
    double scales_w_attr = scales_w.value_or(1);
    auto outputSize = op_infer::upsample_bilinear2d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());

    EXEC_NPU_CMD(aclnnUpsampleBilinear2d, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bilinear2d(
    const at::Tensor& self_ex,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2d,
                     acl_op::upsample_bilinear2d(self_ex, output_size, align_corners, scale_factors));
    at::Tensor self = self_ex;
    auto osize = op_infer::upsample_infershape_with_scale(self_ex.sizes(), output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);

    TORCH_CHECK(self.scalar_type() != at::ScalarType::Double, "upsample_binlinear_2d not support torch.fp64 dtypes",
                OPS_ERROR(ErrCode::TYPE));

    auto output_osize = op_infer::upsample_bilinear2d_npu_output_size(self, osize);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_osize, self.options());

    auto output_osize2 = at::IntArrayRef(osize);
    double scales_h_attr = scales_h.value_or(1);
    double scales_w_attr = scales_w.value_or(1);
    EXEC_NPU_CMD(aclnnUpsampleBilinear2d, self, output_osize2, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}
#endif
}
