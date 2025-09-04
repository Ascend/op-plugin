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
#include "op_plugin/utils/OpAdapter.h"

static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DISABLED = 0;
static const int64_t MODE_TOKEN_COUNT = 2;


namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_init_routing_quant(const at::Tensor &x, const at::Tensor &expert_idx,
        const c10::optional <at::Tensor> &scale, const c10::optional <at::Tensor> &offset,
        int64_t active_num, int64_t expert_capacity, int64_t expert_num,
        int64_t drop_pad_mode, int64_t expert_tokens_num_mode,
        bool expert_tokens_before_capacity_flag, int64_t quant_mode)
    {
        TORCH_CHECK(x.dim() == DIM_TWO, "Input tensor 'x' must be 2-dimensional, but got dimension ", x.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(expert_idx.dim() == DIM_TWO,
            "Input tensor 'expert_idx' must be 2-dimensional, but got dimension ", expert_idx.dim(), OPS_ERROR(ErrCode::PARAM));

        auto x_size = x.sizes();
        auto expert_idx_size = expert_idx.sizes();
        TORCH_CHECK(x_size[0] == expert_idx_size[0],
            "The number of rows in input 'x' (", x_size[0], ") must match the number of rows in 'expert_idx' (", expert_idx_size[0], ").", OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(drop_pad_mode == 0 || drop_pad_mode == 1,
            "Parameter 'drop_pad_mode' must be 0 or 1, but got: ", drop_pad_mode, OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "Parameter 'quant_mode' must be 0 or 1, but got: ", quant_mode, OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(expert_tokens_num_mode >= DISABLED && expert_tokens_num_mode <= MODE_TOKEN_COUNT,
            "Parameter 'expert_tokens_num_mode' must be 0, 1, or 2, but got: ", expert_tokens_num_mode, OPS_ERROR(ErrCode::VALUE));

        const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
        const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });

        int64_t bs = x_size[0];
        int64_t h = x_size[1];
        int64_t k = expert_idx_size[1];

        at::Tensor expanded_x;
        int64_t expanded_scale_len = 0;
        if (drop_pad_mode == 1) { // Drop/Pad
            expanded_x = npu_preparation::apply_tensor_without_format({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
            expanded_scale_len = expert_num * expert_capacity;
        } else { // Dropless / Active
            if (active_num > 0) { // Active
                int64_t num_out_tokens = std::min((int64_t)bs * k, active_num);
                expanded_x = npu_preparation::apply_tensor_without_format({num_out_tokens, h}, x.options().dtype(at::kChar));
                expanded_scale_len = num_out_tokens;
            } else { // Dropless
                expanded_x = npu_preparation::apply_tensor_without_format({bs * k, h}, x.options().dtype(at::kChar));
                expanded_scale_len = bs * k;
            }
        }

        at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format({bs * k}, expert_idx.options());

        at::Tensor expert_token_cumsum_or_count;
        if (drop_pad_mode == 0 && expert_tokens_num_mode > 0) {
            expert_token_cumsum_or_count = npu_preparation::apply_tensor_without_format({expert_num}, x.options().dtype(at::kInt));
        } else {
            expert_token_cumsum_or_count = at::Tensor();
        }

        at::Tensor expert_tokens_before_capacity;
        if (drop_pad_mode == 1 && expert_tokens_before_capacity_flag) {
            expert_tokens_before_capacity = npu_preparation::apply_tensor_without_format({expert_num}, x.options().dtype(at::kInt));
        } else {
            expert_tokens_before_capacity = at::Tensor();
        }

        at::Tensor expanded_scale;
        if (quant_mode == 1) {
            expanded_scale = npu_preparation::apply_tensor_without_format({expanded_scale_len}, x.options().dtype(at::kFloat));
        } else {
            expanded_scale = at::Tensor();
        }

        EXEC_NPU_CMD(aclnnMoeInitRoutingQuantV2, x, expert_idx, p_scale, p_offset,
            active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_mode,
            expert_tokens_before_capacity_flag, quant_mode, expanded_x, expanded_row_idx,
            expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale);

        return std::tie(expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale);
    }
}
