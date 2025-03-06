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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/UpsampleConstants.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

bool checkBilinearScales(float realScale_h, float realScale_w)
{
    return !(realScale_h > MAX_SUPPORT_SCALE || realScale_w > MAX_SUPPORT_SCALE);
}

bool checkBilinearUseFast(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners, double scales_h,
    double scales_w, at::Tensor &result)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                              c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    double realScale_h =
        op_plugin::utils::compute_scale(self.size(H_INDEX), result.size(H_INDEX), scales_h);
    double realScale_w =
        op_plugin::utils::compute_scale(self.size(W_INDEX), result.size(W_INDEX), scales_w);
    if (!is_support_nd_out || !checkBilinearScales(realScale_h, realScale_w)) {
        return false;
    }
    return true;
}

at::Tensor &upsample_bilinear2d_aa_out_slow(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &result)
{
    auto scalar_type = self.scalar_type();
    at::Tensor self_slow = self.cpu().to(at::ScalarType::Float);

    at::Tensor result_slow = at::_upsample_bilinear2d_aa(self_slow, output_size, align_corners, scales_h, scales_w);
    result.copy_(result_slow.to(scalar_type));
    return result;
}

at::Tensor upsample_bilinear2d_aa_slow(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    auto scalar_type = self.scalar_type();
    auto outputSize =
        op_infer::upsample_bilinear2d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
    at::Tensor self_slow = self.cpu().to(at::ScalarType::Float);

    at::Tensor result_slow = at::_upsample_bilinear2d_aa(self_slow, output_size, align_corners, scales_h, scales_w);
    result.copy_(result_slow.to(scalar_type));
    return result;
}

at::Tensor &_upsample_bilinear2d_aa_out(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dAA,
        upsample_bilinear2d_aa_out_slow(self, output_size, align_corners, scales_h, scales_w, result));

    auto outputSize =
        op_infer::upsample_bilinear2d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, result, self, outputSize);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    if (!checkBilinearUseFast(self, output_size, align_corners, scales_h_attr, scales_w_attr, result)) {
        return upsample_bilinear2d_aa_out_slow(self, output_size, align_corners, scales_h, scales_w, result);
    }
    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAA, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

at::Tensor _upsample_bilinear2d_aa(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(
        aclnnUpsampleBilinear2dAA, upsample_bilinear2d_aa_slow(self, output_size, align_corners, scales_h, scales_w));

    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    auto outputSize =
        op_infer::upsample_bilinear2d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());
    if (!checkBilinearUseFast(self, output_size, align_corners, scales_h_attr, scales_w_attr, result)) {
        return upsample_bilinear2d_aa_slow(self, output_size, align_corners, scales_h, scales_w);
    }
    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAA, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

}  // namespace op_api
