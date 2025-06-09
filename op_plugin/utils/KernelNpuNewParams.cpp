// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/utils/KernelNpuNewParams.h"

namespace op_infer {

int64_t npu_gelu_approximate_mode(c10::string_view approximate)
{
    std::string approximate_str = std::string(approximate);
    TORCH_CHECK(approximate_str == "tanh" || approximate_str == "none",
        "NPU error, approximate argument must be either none or tanh.", OPS_ERROR(ErrCode::PARAM));
    int64_t approximate_mode = approximate_str == "tanh" ? 1 : 0;
    return approximate_mode;
}

std::string npu_gelu_approximate_str(c10::string_view approximate)
{
    std::string approximate_str = std::string(approximate);
    TORCH_CHECK(approximate_str == "tanh" || approximate_str == "none",
        "NPU error, approximate argument must be either none or tanh.", OPS_ERROR(ErrCode::PARAM));
    return approximate_str;
}

bool npu_add_rms_norm_quant_param_check(
    c10::optional<at::Tensor> scales2,
    c10::optional<at::Tensor> zero_points2,
    int64_t axis,
    bool div_mode)
{
    TORCH_CHECK(!scales2.has_value(), "scales2 only support None.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!zero_points2.has_value(), "zero_points2 only support None.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(axis == -1, "axis only support -1.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(div_mode == true, "div_mode only support True.", OPS_ERROR(ErrCode::PARAM));
    return true;
}

} // namespace op_infer
