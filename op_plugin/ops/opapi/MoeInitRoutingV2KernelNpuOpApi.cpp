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

static const int64_t DIM_TWO = 2;
static const int64_t EXPERT_RANGE_DIM = 2;
static const int64_t MAX_EXPERT_NUM = 10240;
static const int64_t UN_QUANT = -1;
static const int64_t STATIC_QUANT = 0;
static const int64_t DYNAMIC_QUANT = 1;
static const int64_t GATHER = 0;
static const int64_t SCATTER = 1;
static const int64_t CUMSUM = 0;
static const int64_t COUNT = 1;

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using npu_utils = at_npu::native::NpuUtils;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_init_routing_v2(const at::Tensor &x, const at::Tensor &expert_idx,
                                        const c10::optional <at::Tensor> &scale,
                                        const c10::optional <at::Tensor> &offset,
                                        int64_t active_num, int64_t expert_capacity, int64_t expert_num,
                                        int64_t drop_pad_mode, int64_t expert_tokens_num_type,
                                        bool expert_tokens_num_flag, int64_t quant_mode,
                                        at::IntArrayRef active_expert_range, int64_t row_idx_type)
    {
        TORCH_CHECK(x.dim() == DIM_TWO, "The x should be 2D");
        TORCH_CHECK(x.scalar_type() == at::kChar || x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
                    x.scalar_type() == at::kBFloat16,
                    "an int8, float16, float32 or bfloat16 tensor is expected but got a tensor with dtype: ",
                    x.scalar_type());
        TORCH_CHECK(expert_idx.dim() == DIM_TWO, "The expert_idx should be 2D");
        TORCH_CHECK(expert_idx.scalar_type() == at::kInt,
                    "int32 tensor expected but got a tensor with dtype: ",
                    expert_idx.scalar_type());

        TORCH_CHECK(!active_expert_range.empty(), "active_expert_range is required.");
        TORCH_CHECK(active_expert_range.size() == EXPERT_RANGE_DIM, "active_expert_range must have 2 element");
        TORCH_CHECK(active_expert_range[0] < active_expert_range[1], "Invalid active_expert_range: must be increasing",
                    active_expert_range[0], " < ", active_expert_range[1]);
        TORCH_CHECK(active_expert_range[1] <= MAX_EXPERT_NUM && active_expert_range[0] >= 0,
                    "active_expert_range must be within [0, 10240]");

        TORCH_CHECK(drop_pad_mode == 0 || drop_pad_mode == 1, "drop_pad_mode must be in [0, 1]");
        TORCH_CHECK(expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT,
                    "expert_tokens_num_type must be in [0, 1]");
        TORCH_CHECK(quant_mode >= UN_QUANT && quant_mode <= DYNAMIC_QUANT, "quant_mode must be in [-1, 0, 1]");
        TORCH_CHECK(row_idx_type == GATHER || row_idx_type == SCATTER, "row_idx_type must be in [0, 1]");

        int expert_length = active_expert_range[1] - active_expert_range[0];
        auto x_size = x.sizes();
        auto expert_idx_size = expert_idx.sizes();
        TORCH_CHECK(x_size[0] == expert_idx_size[0], "Input rows of x and expert_idx should be same.");

        const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
        const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });
        if (p_scale.defined()) {
            TORCH_CHECK(p_scale.scalar_type() == at::kFloat, "float32 tensor expected but got a tensor with dtype: ",
                        p_scale.scalar_type());
            auto scale_size = p_scale.sizes();
            if (quant_mode == -1) {
                // no quant mode, scale should be (bs,)
                TORCH_CHECK(p_scale.dim() == 1, "The scale should be 1D");
                TORCH_CHECK(x_size[0] == scale_size[0],
                            "Input rows of scale should be the same with the length of x[0].");
            } else if (quant_mode == 0) {
                TORCH_CHECK(x.scalar_type() != at::kChar,
                            "tensor x has dtype int8 which is not supported in static/dynamic quaant mode");
                // static quant mode, scale should be (end-start, h) or (end-start, 1)
                TORCH_CHECK(p_scale.dim() == DIM_TWO || p_scale.dim() == 1,
                            "The scale should be (end-start, 1) or (end-start,)");
                TORCH_CHECK(expert_length == scale_size[0],
                            "Input rows of scale should be the same with the length of active_expert_range.");
                TORCH_CHECK(p_scale.dim() == 1 || x_size[1] == scale_size[1] || 1 == scale_size[1],
                            "Input cols of scale should be 1 or the same with the input cols of x.");

                if (p_offset.defined()) {
                    // offset is checked only in static quant mode and only when scale is checked, dim/shapes are the same with scale, if not None
                    TORCH_CHECK(p_offset.scalar_type() == at::kFloat,
                                "float32 tensor expected but got a tensor with dtype: ", p_offset.scalar_type());
                    TORCH_CHECK(p_offset.dim() == DIM_TWO || p_offset.dim() == 1,
                                "The offset should (end-start, 1) or (end-start,)");
                    auto offset_size = p_offset.sizes();
                    TORCH_CHECK(scale_size[0] == offset_size[0],
                                "Input rows of offset should be the same with the length of active_expert_range.");
                    TORCH_CHECK(p_offset.dim() == 1 || x_size[1] == offset_size[1] || 1 == offset_size[1],
                                "Input cols of offset should be the same with the input cols of scale_size.");
                }
            } else {
                TORCH_CHECK(x.scalar_type() != at::kChar,
                            "tensor x has dtype int8 which is not supported in static/dynamic quaant mode");
                // dynamic quant mode, scale should be (end-start, h)
                TORCH_CHECK(p_scale.dim() == DIM_TWO, "The scale should be 2D");
                TORCH_CHECK(expert_length == scale_size[0],
                            "Input rows of scale should be the same with the length of active_expert_range.");
                TORCH_CHECK(x_size[1] == scale_size[1],
                            "Input cols of scale should be the same with the input cols of x.");
            }
        }

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
        at::Tensor expert_tokens_count_or_cumsum = npu_preparation::apply_tensor_without_format({expert_length},
                                                                                                x.options().dtype(
                                                                                                    at::kLong));
        at::Tensor expanded_scale = npu_preparation::apply_tensor_without_format({bs * k},
                                                                                 x.options().dtype(at::kFloat));
        EXEC_NPU_CMD(aclnnMoeInitRoutingV3, x, expert_idx, p_scale, p_offset,
                     active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
                     expert_tokens_num_flag,
                     quant_mode, active_expert_range, row_idx_type,
                     expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
        return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
    }
}
