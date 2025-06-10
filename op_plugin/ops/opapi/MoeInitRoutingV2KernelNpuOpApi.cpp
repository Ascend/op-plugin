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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

// expert_tokens_num_types
// 'expert_tokens_count_or_cumsum' is suggested to be reverted into 'expert_tokens_count'
static const int64_t CUMSUM = 0;
static const int64_t COUNT = 1;
static const int64_t KEY_VALUE = 2;

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

tensor_list npu_moe_init_routing_v2(const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type)
{
    int expert_length = active_expert_range[1] - active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
    const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });

    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];
    at::Tensor expanded_x;
    if (quant_mode == -1) {
        expanded_x = npu_preparation::apply_tensor_without_format(x, {bs * k, h});
    } else {
        expanded_x = npu_preparation::apply_tensor_without_format({bs * k, h}, x.options().dtype(at::kChar));
    }
    at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format(expert_idx, {bs * k});
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum =
            npu_preparation::apply_tensor_without_format({expert_length}, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum =
            npu_preparation::apply_tensor_without_format({expert_num, 2}, x.options().dtype(at::kLong));
    }

    at::Tensor expanded_scale = npu_preparation::apply_tensor_without_format({bs * k}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMoeInitRoutingV3,
        x,
        expert_idx,
        p_scale,
        p_offset,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range,
        row_idx_type,
        expanded_x,
        expanded_row_idx,
        expert_tokens_count_or_cumsum,
        expanded_scale);
    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}
}  // namespace op_api
