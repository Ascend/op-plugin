// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

at::Tensor npu_trans_quant_param(const at::Tensor &scale, const c10::optional<at::Tensor> &offset,
                                 c10::optional<int64_t> round_mode)
{
    auto scale_dim_num = scale.dim();
    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    auto output_size = op_infer::array_to_small_vector(scale.sizes());
    // infer out shape for the case that scale dim equals to one
    if (scale.dim() == 1 && offset.has_value()) {
        output_size = op_infer::array_to_small_vector((scale.size(0) > offset_real.size(0)) ?
                                                       scale.sizes() : offset_real.sizes());
    }
    int64_t round_mode_value = round_mode.value_or(0);
    TORCH_CHECK(round_mode_value == 0 || round_mode_value == 1, "round_mode must be 0 or 1. but now is ",
                round_mode_value);
    c10::TensorOptions options = scale.options().dtype(at::kLong);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);
    static const bool is_trans_quant_param_V3_available = check_aclnn_kernel_available("aclnnTransQuantParamV3");
    if (!is_trans_quant_param_V3_available) {
        TORCH_CHECK(round_mode_value == 0, "aclnnTransQuantParamV2 can't support round_mode, please upgrade CANN.")
        EXEC_NPU_CMD(aclnnTransQuantParamV2, scale, offset_real, result);
    } else {
        EXEC_NPU_CMD(aclnnTransQuantParamV3, scale, offset_real, round_mode_value, result);
    }
    return result;
}
}