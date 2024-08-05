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

at::Tensor& upsample_bicubic2d_opapi(const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
                                     c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor& result)
{
    double scales_h_value = scales_h.value_or(0);
    double scales_w_value = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleBicubic2d, self, output_size, align_corners, scales_h_value, scales_w_value, result);
    return result;
}

at::Tensor& upsample_bicubic2d_out(const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
                                   c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2d,
                     acl_op::upsample_bicubic2d_out(self, output_size, align_corners, scales_h, scales_w, result));
    auto outputsize = op_infer::upsample_bicubic2d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, result, self, outputsize);
    upsample_bicubic2d_opapi(self, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

at::Tensor upsample_bicubic2d(const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
                              c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2d,
                     acl_op::upsample_bicubic2d(self, output_size, align_corners, scales_h, scales_w));
    auto outputsize = op_infer::upsample_bicubic2d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputsize);
    upsample_bicubic2d_opapi(self, output_size, align_corners, scales_h, scales_w, result);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bicubic2d(const at::Tensor& self, c10::optional<at::IntArrayRef> output_size,
                              bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2d,
                     acl_op::upsample_bicubic2d(self, output_size, align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    at::Tensor result = op_api::upsample_bicubic2d(self, osize, align_corners, scales_h, scales_w);
    return result;
}
#endif

} // namespace op_api
