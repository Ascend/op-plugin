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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::SmallVector<int64_t, SIZE> upsample_nearest_exact2d_output_size_npu(
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

at::Tensor &_upsample_nearest_exact2d_out(const at::Tensor &self, at::IntArrayRef output_size,
    c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &result)
{
    auto outputSize = upsample_nearest_exact2d_output_size_npu(self, output_size);
    npu_preparation::check_tensor({self}, result, self, outputSize);
    double scalesH = scales_h.value_or(0);
    double scalesW = scales_w.value_or(0);

    EXEC_NPU_CMD(aclnnUpsampleNearestExact2d, self, output_size, scalesH, scalesW, result);
    return result;
}

at::Tensor _upsample_nearest_exact2d(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    double scalesH = scales_h.value_or(0);
    double scalesW = scales_w.value_or(0);
    auto outputSize = upsample_nearest_exact2d_output_size_npu(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());

    EXEC_NPU_CMD(aclnnUpsampleNearestExact2d, self, output_size, scalesH, scalesW,  result);
    return result;
}

}  // namespace op_api
