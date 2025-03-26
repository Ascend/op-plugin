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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::SmallVector<int64_t, SIZE> upsample_nearest2d_output_size_npu(
    const at::Tensor &input,
    at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() >= 2, "Input's dim must be at least 2.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() >= 2, "Output size must be at least 2.", OPS_ERROR(ErrCode::PARAM));
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    at::SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};
    return outputSize;
}

at::Tensor &upsample_nearest2d_old_out(
    const at::Tensor &self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest2d,
                     acl_op::upsample_nearest2d_out(self, output_size, scales_h, scales_w, out));
    at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
    npu_preparation::check_tensor({self}, out, self, outputSize);
    EXEC_NPU_CMD(aclnnUpsampleNearest2d, self, output_size, out);
    return out;
}

at::Tensor upsample_nearest2d_old(
    const at::Tensor &self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest2d, acl_op::upsample_nearest2d(self, output_size, scales_h, scales_w));
    at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
    at::Tensor out = npu_preparation::apply_tensor_without_format(self, outputSize);
    EXEC_NPU_CMD(aclnnUpsampleNearest2d, self, output_size, out);
    return out;
}

at::Tensor &upsample_nearest2d_out(
    const at::Tensor &self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest2dV2,
                     op_api::upsample_nearest2d_old_out(self, output_size, scales_h, scales_w, out));
    at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
    npu_preparation::check_tensor({self}, out, self, outputSize);
    float scale_h = static_cast<float>(scales_h.value_or(-1.0));
    float scale_w = static_cast<float>(scales_w.value_or(-1.0));
    EXEC_NPU_CMD(aclnnUpsampleNearest2dV2, self, output_size, scale_h, scale_w, out);
    return out;
}

at::Tensor upsample_nearest2d(
    const at::Tensor &self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest2dV2, op_api::upsample_nearest2d_old(self, output_size, scales_h, scales_w));
    float scale_h = static_cast<float>(scales_h.value_or(-1.0));
    float scale_w = static_cast<float>(scales_w.value_or(-1.0));
    at::SmallVector<int64_t, SIZE> outputSize = upsample_nearest2d_output_size_npu(self, output_size);
    at::Tensor out = npu_preparation::apply_tensor_without_format(self, outputSize);
    EXEC_NPU_CMD(aclnnUpsampleNearest2dV2, self, output_size, scale_h, scale_w, out);
    return out;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest2d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    auto osize = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
    auto scale_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scale_w = op_plugin::utils::get_scale_value(scale_factors, 1);

    return op_api::upsample_nearest2d(input, osize, scale_h, scale_w);
}
#endif

}
