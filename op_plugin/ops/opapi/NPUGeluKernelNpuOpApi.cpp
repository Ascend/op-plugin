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
constexpr static int64_t APPROXIMATE_TANH = 1;

at::Tensor npu_gelu(const at::Tensor &self, c10::string_view approximate)
{
    std::string approximate_str = std::string(approximate);
    TORCH_CHECK(approximate_str == "tanh" || approximate_str == "none",
        "NPU error, approximate argument must be either none or tanh.", OPS_ERROR(ErrCode::PARAM));
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);

    int64_t approximate_mode = 0;
    if (approximate_str == "tanh") {
        approximate_mode = APPROXIMATE_TANH;
    }
    EXEC_NPU_CMD(aclnnGeluV2, self, approximate_mode, result);
    return result;
}
}
