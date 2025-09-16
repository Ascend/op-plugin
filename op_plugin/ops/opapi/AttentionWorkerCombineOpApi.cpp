// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor> npu_attention_worker_combine(
        const at::Tensor &schedule_context,
        const at::Tensor &expert_scales,
        const at::Tensor &layer_id,
        int64_t hidden_size,
        int64_t token_dtype,
        int64_t need_schedule)
    {
        at::SmallVector <int64_t, op_infer::SIZE> y_size;
        at::SmallVector <int64_t, op_infer::SIZE> next_layer_id_size;
 
        y_size.push_back(expert_scales.size(0));
        y_size.push_back(hidden_size);
        next_layer_id_size.push_back(1);

        auto y_dtype = at::kHalf;
        if (token_dtype == 1) {
            y_dtype = at::kBFloat16;
        }

        at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, schedule_context.options().dtype(y_dtype));
        at::Tensor next_layer_id = npu_preparation::apply_tensor_without_format(next_layer_id_size, schedule_context.options().dtype(at::kInt));

        EXEC_NPU_CMD(aclnnAttentionWorkerCombine, schedule_context, expert_scales, layer_id,
            hidden_size, token_dtype, need_schedule,
            y, next_layer_id);

        return std::tie(y, next_layer_id);
    }
}