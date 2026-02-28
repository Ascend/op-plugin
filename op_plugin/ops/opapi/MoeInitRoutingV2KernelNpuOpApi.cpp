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

namespace {
constexpr int64_t DIM_X = 2;
constexpr int64_t DIM_EXPERT_IDX = 2;
constexpr int64_t LENGTH_ACTIVE_EXPERT_RANGE = 2;
constexpr int64_t EXPERT_TOKENS_COUNT = 1;
constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
constexpr int64_t QUANT_MODE_UNQUANT = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_HIF8_PERTENSOR = 7;
constexpr int64_t QUANT_MODE_HIF8_PER_TOKEN_DIM = 8;
constexpr int64_t MXQUANT_BLOCK_SIZE = 32;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;

constexpr int64_t EXPERT_NUM_V2 = 128;
constexpr int64_t EXPERT_NUM_MIN_V2 = 0;
constexpr int64_t EXPERT_NUM_MAX_V2 = 128;
constexpr int64_t HIDDEN_DIM_VAL_V2 = 2048;
};  // namespace

inline bool IsQuantModeMXFP8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN;
}

inline bool IsQuantModeHIF8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_HIF8_PERTENSOR || quantMode == QUANT_MODE_HIF8_PER_TOKEN_DIM;
}

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

at::IntArrayRef init_new_active_expert_range(at::IntArrayRef &active_expert_range, int64_t expert_num)
{
    if (active_expert_range.empty()) {
        static std::vector<int64_t> default_active_expert_range = {0, expert_num};
        return at::IntArrayRef(default_active_expert_range);
    } else {
        return active_expert_range;
    }
}

static bool CheckV2Case(int hidden_dim, int64_t expert_num, at::IntArrayRef active_expert_range,
    int64_t expert_tokens_num_type, int64_t quant_mode)
{
    if (expert_num == EXPERT_NUM_V2 && active_expert_range[0] == EXPERT_NUM_MIN_V2 &&
        active_expert_range[1] == EXPERT_NUM_MAX_V2 && hidden_dim == HIDDEN_DIM_VAL_V2) {
        if (quant_mode == -1 && expert_tokens_num_type == 1) {
            return true;
        }
    }
    return false;
}

tensor_list npu_moe_init_routing_v2(const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type)
{
#if !VERSION_BETWEEN(V2R7, VERSION_NEWEST)
    // 小于2.7的版本不支持MXFP8量化需要的float8_e8m0fnu类型
    TORCH_CHECK(!IsQuantModeMXFP8(quant_mode),
        "Unsupported quant_mode:",
        quant_mode,
        " on this version of torch with torch_npu. Please upgrade to at least v2.7.",
        OPS_ERROR(ErrCode::PARAM));
#endif

    int64_t x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_X,
        "The x should be ",
        DIM_X,
        "-Dimension, current is ",
        x_dim,
        "-Dimension.",
        OPS_ERROR(ErrCode::PARAM));

    int64_t expert_idx_dim = expert_idx.dim();
    TORCH_CHECK(expert_idx_dim == DIM_EXPERT_IDX,
        "The expert_idx should be ",
        DIM_EXPERT_IDX,
        "-Dimension, current is ",
        expert_idx,
        "-Dimension.",
        OPS_ERROR(ErrCode::PARAM));

    at::IntArrayRef current_active_expert_range = init_new_active_expert_range(active_expert_range, expert_num);
    int64_t active_expert_range_length = current_active_expert_range.size();
    TORCH_CHECK(active_expert_range_length == LENGTH_ACTIVE_EXPERT_RANGE,
        "The length of list active_expert_range should be ",
        LENGTH_ACTIVE_EXPERT_RANGE,
        ", current is ",
        active_expert_range_length,
        ".",
        OPS_ERROR(ErrCode::PARAM));

    int expert_length = current_active_expert_range[1] - current_active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
    const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });

    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];
    // more suitable cases for v2
    bool using_v2 = CheckV2Case(h, expert_num, active_expert_range, expert_tokens_num_type, quant_mode);

    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;
    if (drop_pad_mode == 1) {  // Drop/Pad
        if (quant_mode == QUANT_MODE_UNQUANT) {
            expanded_x = npu_preparation::apply_tensor_without_format(x, {expert_num, expert_capacity, h});
        } else {
            expanded_x = npu_preparation::apply_tensor_without_format(
                {expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
        }
        expanded_scale_len = expert_num * expert_capacity;
    } else {  // Dropless
        expanded_scale_len = (active_num <= 0) ? bs * k : std::min<int64_t>(active_num, bs * k);
        switch (quant_mode) {
#if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
            case QUANT_MODE_MXFP8_E5M2:
                expanded_x = npu_preparation::apply_tensor_without_format(
                    {expanded_scale_len, h}, x.options().dtype(at::kFloat8_e5m2));
                break;
            case QUANT_MODE_MXFP8_E4M3FN:
                expanded_x = npu_preparation::apply_tensor_without_format(
                    {expanded_scale_len, h}, x.options().dtype(at::kFloat8_e4m3fn));
                break;
#endif
            case QUANT_MODE_STATIC:
            case QUANT_MODE_DYNAMIC:
                expanded_x =
                    npu_preparation::apply_tensor_without_format({expanded_scale_len, h}, x.options().dtype(at::kChar));
                break;
            case QUANT_MODE_HIF8_PERTENSOR:
            case QUANT_MODE_HIF8_PER_TOKEN_DIM:
                expanded_x =
                    npu_preparation::apply_tensor_without_format({expanded_scale_len, h}, x.options().dtype(at::kByte));
                break;
            default:  // quant_mode == QUANT_MODE_UNQUANT
                expanded_x = npu_preparation::apply_tensor_without_format(x, {expanded_scale_len, h});
        }
    }

    at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format(expert_idx, {bs * k});
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type < EXPERT_TOKENS_KEY_VALUE) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum =
            npu_preparation::apply_tensor_without_format({expert_length}, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == EXPERT_TOKENS_KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum =
            npu_preparation::apply_tensor_without_format({expert_num, 2}, x.options().dtype(at::kLong));
    }

    if (using_v2 && !op_plugin::utils::is_gte_cann_version_850alpha003()) {
        at::Tensor expert_tokens_before_capacity =
            npu_preparation::apply_tensor_without_format({expert_num}, x.options().dtype(at::kInt));
        expert_capacity = 0;
        drop_pad_mode = 0;
        int64_t expert_tokens_count_or_cumsum_flag = 2;
        bool expert_tokens_before_capacity_flag = false;
        EXEC_NPU_CMD(aclnnMoeInitRoutingV2,
            x,
            expert_idx,
            active_num,
            expert_capacity,
            expert_num,
            drop_pad_mode,
            expert_tokens_count_or_cumsum_flag,
            expert_tokens_before_capacity_flag,
            expanded_x,
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity);
        return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity);
    }

#if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
    at::Tensor expanded_scale;
    if (IsQuantModeMXFP8(quant_mode)) {
        // scale_cols为h向上整除32后向上对齐到偶数倍
        int64_t scale_cols = (h + MXQUANT_BLOCK_SIZE - 1) / MXQUANT_BLOCK_SIZE;
        scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) / PAD_TO_EVEN_FACTOR * PAD_TO_EVEN_FACTOR;
        expanded_scale = npu_preparation::apply_tensor_without_format(
            {expanded_scale_len, scale_cols}, x.options().dtype(at::kFloat8_e8m0fnu));
    } else {
        expanded_scale =
            npu_preparation::apply_tensor_without_format({expanded_scale_len}, x.options().dtype(at::kFloat));
    }
#else
    at::Tensor expanded_scale =
        npu_preparation::apply_tensor_without_format({expanded_scale_len}, x.options().dtype(at::kFloat));
#endif
    TensorWrapper expanded_x_wrapper = {expanded_x, npu_preparation::convert_to_acl_data_type(expanded_x.scalar_type())};
    if (IsQuantModeHIF8(quant_mode)) {
        expanded_x_wrapper.dtype = aclDataType::ACL_HIFLOAT8;
    }
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
        expanded_x_wrapper,
        expanded_row_idx,
        expert_tokens_count_or_cumsum,
        expanded_scale);
    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}
}  // namespace op_api