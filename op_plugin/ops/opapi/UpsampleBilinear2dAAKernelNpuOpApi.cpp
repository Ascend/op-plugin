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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &_upsample_bilinear2d_aa_out(const at::Tensor &self_ex, at::IntArrayRef output_size,
                                        bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &result)
{
    at::Tensor self = self_ex;
    auto outputSize = op_infer::upsample_bilinear2d_npu_output_size(self, output_size, align_corners, scales_h, scales_w);
    npu_preparation::check_tensor({self}, result, self, outputSize);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAA, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

at::Tensor _upsample_bilinear2d_aa(const at::Tensor &self_ex, at::IntArrayRef output_size,
                                   bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    at::Tensor self = self_ex;
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    auto outputSize = op_infer::upsample_bilinear2d_npu_output_size(self, output_size, align_corners, scales_h, scales_w);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dAA, self, output_size, align_corners, scales_h_attr, scales_w_attr, result);
    return result;
}

} // namespace op_api
