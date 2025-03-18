// Copyright (c) 2024-2025 Huawei Technologies Co., Ltd
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

at::Tensor &upsample_nearest_exact1d_out_slow(const at::Tensor &self, at::IntArrayRef output_size,
    c10::optional<double> scales, at::Tensor &result)
{
    at::Tensor result_slow = at::_upsample_nearest_exact1d(self.cpu(), output_size, scales);
    result.copy_(result_slow);
    return result;
}

at::Tensor upsample_nearest_exact1d_slow(const at::Tensor &self, at::IntArrayRef output_size,
    c10::optional<double> scales)
{
    auto outputSize = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());

    at::Tensor result_slow = at::_upsample_nearest_exact1d(self.cpu(), output_size, scales);
    result.copy_(result_slow);
    return result;
}

at::Tensor &_upsample_nearest_exact1d_out(const at::Tensor &self, at::IntArrayRef output_size,
    c10::optional<double> scales, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnUpsampleNearestExact1d, upsample_nearest_exact1d_out_slow(self, output_size, scales, out));

    auto outputSize = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, out, self, outputSize);
    double scalesValue = scales.value_or(0);

    EXEC_NPU_CMD(aclnnUpsampleNearestExact1d, self, output_size, scalesValue, out);
    return out;
}

at::Tensor _upsample_nearest_exact1d(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleNearestExact1d, upsample_nearest_exact1d_slow(self, output_size, scales));

    double scalesValue = scales.value_or(0);
    auto outputSize = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options());

    EXEC_NPU_CMD(aclnnUpsampleNearestExact1d, self, output_size, scalesValue,  result);
    return result;
}

}  // namespace op_api
