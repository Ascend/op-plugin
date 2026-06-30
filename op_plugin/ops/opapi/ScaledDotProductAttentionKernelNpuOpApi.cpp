// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const static int64_t ATTENMASK_LIMIT = 2048;
const static int64_t B_LIMIT = 65536;
const static int64_t N_LIMIT = 2048;
const static int64_t FIA_N_LIMIT = 256;
const static int64_t D_LIMIT = 512;
const static int64_t BNSD_DIM = 4;
const static int64_t TOKEN_MAX = 2147483647;
const static int64_t LEFT_UP_CAUSAL = 2;
const static int64_t ATTN_MASK_DIM_TWO = 2;
const static int64_t ATTN_MASK_DIM_FOUR = 4;


inline void validate_sdpa_input(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask)
{
    TORCH_CHECK(query.dtype() == key.dtype() && query.dtype() == value.dtype(),
        "Expected query, key, and value to have the same dtype, but got query.dtype: ",
        query.dtype(), " key.dtype: ", key.dtype(), " and value.dtype: ", value.dtype(), " instead." +
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
        "Expected query, key, and value to have the same device type, but got query.device: ",
        query.device(), " key.device: ", key.device(), " and value.device: ", value.device(), " instead." +
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(query.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
        "Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: ",
        query.dim(), " key.dim: ", key.dim(), " and value.dim: ", value.dim(), " instead." +
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    if (attn_mask.has_value()) {
        auto mask_dtype = attn_mask->dtype();
        TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == query.dtype(),
            "Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: ",
            mask_dtype, " and  query.dtype: ", query.dtype(), " instead." + OPS_ERROR(ErrCode::NOT_SUPPORT));
        TORCH_CHECK(!query.is_nested() && !key.is_nested(),
            "Scaled_dot_product_attention: Nested tensors for query / key are not supported "
            "when an explicit attn_mask is set" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    return;
}

c10::optional<at::Tensor> convert_boolean_attn_mask_math(
    const c10::optional<at::Tensor> &attn_mask,
    caffe2::TypeMeta dtype)
{
    if (!attn_mask.has_value()) {
        return c10::nullopt;
    }
    if (attn_mask->dtype() == at::kBool) {
        auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
        new_attn_mask.masked_fill_(attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
        return new_attn_mask;
    }
    return attn_mask;
}

c10::optional<at::Tensor> convert_boolean_attn_mask(
    const at::Tensor &query,
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal)
{
    if (!attn_mask.has_value() && !is_causal) {
        return c10::nullopt;
    }
    if (is_causal) {
        TORCH_CHECK(!attn_mask.has_value(),
            "The attn_mask should be none when is_causal is true, but got ",
            attn_mask.has_value(), "-value");
        at::Tensor atten_mask_shape = at::ones({ATTENMASK_LIMIT, ATTENMASK_LIMIT}, query.options().dtype(at::kBool));
        auto new_attn_mask = at::triu(atten_mask_shape, 1);
        return new_attn_mask;
    }
    const at::Tensor &atten_mask_in = attn_mask.value_or(at::Tensor());
    at::Tensor atten_mask = at::logical_not(atten_mask_in);
    return atten_mask;
}

inline c10::SymFloat calculate_scale(
    const at::Tensor &query,
    c10::optional<double> scale)
{
    const auto softmax_scale = scale.has_value()
        ? scale.value()
        : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
    return c10::SymFloat(softmax_scale);
}

static bool can_broadcast_to(int64_t mask_size, int64_t target_size)
{
    return mask_size == target_size || mask_size == 1;
}

static bool can_expand_attn_mask_for_npu(
    const at::Tensor &mask,
    const at::Tensor &query,
    const at::Tensor &key)
{
    int64_t B = query.size(0), N = query.size(1), Sq = query.size(2), Skv = key.size(2);
    int64_t target_shape[ATTN_MASK_DIM_FOUR] = {B, N, Sq, Skv};
    if (mask.dim() < ATTN_MASK_DIM_TWO || mask.dim() > ATTN_MASK_DIM_FOUR) {
        return false;
    }
    int64_t target_offset = ATTN_MASK_DIM_FOUR - mask.dim();
    for (int64_t i = 0; i < mask.dim(); ++i) {
        if (!can_broadcast_to(mask.size(i), target_shape[target_offset + i])) {
            return false;
        }
    }
    return true;
}

static c10::optional<at::Tensor> preprocess_attn_mask_for_npu(
    const c10::optional<at::Tensor> &attn_mask,
    const at::Tensor &query,
    const at::Tensor &key)
{
    if (!attn_mask.has_value()) {
        return c10::nullopt;
    }
    auto mask = attn_mask.value();
    int64_t B = query.size(0), N = query.size(1), Sq = query.size(2), Skv = key.size(2);

    // Only pre-expand 2D/3D/4D masks known to broadcast to [B, N, Sq, Skv].
    // Other masks are passed through so aclnn keeps the original behavior.
    if (!can_expand_attn_mask_for_npu(mask, query, key)) {
        return mask;
    }

    bool supported_shape = false;
    if (mask.dim() == ATTN_MASK_DIM_TWO) {
        supported_shape = mask.size(0) == Sq && mask.size(1) == Skv;
    } else if (mask.dim() == ATTN_MASK_DIM_FOUR) {
        supported_shape = mask.size(2) == Sq && mask.size(3) == Skv &&
            ((mask.size(0) == B && (mask.size(1) == N || mask.size(1) == 1)) ||
             (mask.size(0) == 1 && mask.size(1) == 1));
    }
    if (supported_shape) {
        return mask;
    }
    return mask.expand({B, N, Sq, Skv});
}

static int64_t calculate_inner_precise_for_fa(
    const at::Tensor &query,
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal)
{
    if (!is_causal && query.size(2) > 1 && attn_mask.has_value()) {
        return 2;
    }
    return 0;
}
#endif

#if VERSION_BETWEEN(V2R1, V2R4)
at::Tensor scaled_dot_product_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale)
{
    validate_sdpa_input(query, key, value, attn_mask);

    static auto compatible_impl = at_npu::native::env::CheckCompatibleImpl();
    bool force_math = false;
    if (compatible_impl) {
        auto& ctx = at::globalContext();
        force_math = ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP();
        if (query.scalar_type() == at::kFloat &&
            (query.size(-1) % 4 != 0 || key.size(-1) % 4 != 0 || value.size(-1) % 4 != 0)) {
            force_math = true;
        }
    }

    if (!force_math &&
        (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat) &&
        ((attn_mask.has_value() && attn_mask->dtype() == at::kBool) || !attn_mask.has_value()) &&
        query.dim() == BNSD_DIM && key.dim() == BNSD_DIM && value.dim() == BNSD_DIM &&
        query.size(1) <= N_LIMIT && query.size(3) <= D_LIMIT && key.size(1) <= N_LIMIT &&
        query.size(1) % key.size(1) == 0 && query.size(1) / key.size(1) > 0 &&
        c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
        auto processed_mask = preprocess_attn_mask_for_npu(
            attn_mask, query, key);

        /* The implementation of the NPU FlashAttention fusion operator without grad constraints:
        1. The FA operator must be registered. GetSocVersion api is provisional, IsExistOp aclnn api will apply.
        2. The attn_mask supports only the bool data type.
        3. The shape [B, N1, S1, D] of the query is suppported, where N1 <= N_LIMIT,
            D <= D_LIMIT and dim == BNSD_DIM.
        4. For GQA, the key shape is [B, N2, S2, D], where N2 <= N_LIMIT, and N1 is a positive integer
            multiple of N2. */
        c10::optional<at::Tensor> atten_mask = convert_boolean_attn_mask(query, processed_mask, is_causal);
        int64_t head_num = query.size(1);
        c10::string_view input_layout = "BNSD";
        auto input_scale = calculate_scale(query, scale);
        double keep_prob = 1 - dropout_p;
        int64_t next_tockens = is_causal ? 0 : TOKEN_MAX;
        int64_t sparse_mode = is_causal ? LEFT_UP_CAUSAL : 0;
        c10::optional<at::Tensor> nulltensor = c10::nullopt;
        c10::OptionalIntArrayRef nulllen = c10::nullopt;
        int64_t inner_precise = calculate_inner_precise_for_fa(query, attn_mask, is_causal);
        auto output =
            at_npu::native::custom_ops::npu_fusion_attention(query, key, value, head_num, input_layout, nulltensor,
                                                             nulltensor, atten_mask, input_scale.as_float_unchecked(),
                                                             keep_prob, TOKEN_MAX, next_tockens, inner_precise, nulllen,
                                                             nulllen, nulllen, sparse_mode, true, false);
        return std::get<0>(output);
    } else if (!force_math &&
               (!query.requires_grad() && !key.requires_grad() && !value.requires_grad()) &&
                (query.dim() == BNSD_DIM && key.dim() == BNSD_DIM && value.dim() == BNSD_DIM) &&
                (query.size(0) != 0 && query.size(1) != 0 && query.size(2) != 0 && query.size(3) != 0) &&
                (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16) && query.size(3) % 16 == 0 &&
                ((attn_mask.has_value() && attn_mask->dtype() == at::kBool && attn_mask->size(-2) == query.size(2) &&
                attn_mask->size(-1) == key.size(2)) || !attn_mask.has_value()) &&
                (query.size(0) == key.size(0) && query.size(3) == key.size(3) && key.size(0) == value.size(0) &&
                key.size(1) == value.size(1) && key.size(2) == value.size(2) && key.size(3) == value.size(3)) &&
                (query.size(0) <= B_LIMIT && query.size(1) <= FIA_N_LIMIT && query.size(3) <= D_LIMIT && key.size(1) <= FIA_N_LIMIT) &&
                (key.size(1) != 0 && query.size(1) % key.size(1) == 0 && query.size(1) / key.size(1) <= 64) &&
                c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
                /* The implementation of the NPU Fused Infer Attention operator constraints:
                1. It only supports the type of fp16 or bf16, and dim = BNSD_DIM
                2. The shape [B, N1, S1, D] of the query is suppported, where B <= B_LIMIT, N1 <= FIA_N_LIMIT,
                    D <= D_LIMIT and must be a positive integer multiple of 16.
                3. The shape of the key and value should be the same, and dim = BNSD_DIM. For GQA, the key shape is [B, N2, S2, D],
                    where 0 < N2 <= FIA_N_LIMIT, N1 is a positive integer multiple of N2 and N1/N2 <= 64.
                4. The attn_mask supports only the bool data type, and shpae[-2] = S1, shpae[-1] = S2.
                5. It only supports SocVersion after Ascend910B1. */
                int64_t inner_precise = 0;
                if (query.size(2) == 1) {
                    is_causal = false;
                } else {
                    if (!is_causal) {
                        inner_precise = 2;
                    }
                }
                auto processed_mask_fia = preprocess_attn_mask_for_npu(
                    attn_mask, query, key);
                c10::optional<at::Tensor> atten_mask = convert_boolean_attn_mask(query, processed_mask_fia, is_causal);
                int64_t head_num = query.size(1);
                int64_t head_num_kv = key.size(1);
                c10::string_view input_layout = "BNSD";
                auto input_scale = calculate_scale(query, scale);
                int64_t next_tockens = is_causal ? 0 : TOKEN_MAX;
                int64_t sparse_mode = is_causal ? LEFT_UP_CAUSAL : 0;
                c10::optional<at::Tensor> nulltensor = c10::nullopt;
                at::OptionalSymIntArrayRef nulllen = c10::nullopt;
                auto output =
                    at_npu::native::custom_ops::npu_fused_infer_attention_score(query, key, value, nulltensor, atten_mask, nulllen, nulllen, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulltensor, nulltensor, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulltensor, nulltensor, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulllen, nulltensor, nulltensor, nulltensor, head_num, input_scale.as_float_unchecked(),
                                                                                TOKEN_MAX, next_tockens, input_layout, head_num_kv, sparse_mode, inner_precise, 0, 0, 0, 0, false);

                return std::get<0>(output);
    }
    c10::optional<at::Tensor> atten_mask_math = convert_boolean_attn_mask_math(attn_mask, query.dtype());
    auto output = at::_scaled_dot_product_attention_math(query, key, value, atten_mask_math, dropout_p, is_causal,
                                                         c10::nullopt, scale);
    return std::get<0>(output);
}
#endif

#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
at::Tensor scaled_dot_product_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale,
    bool enable_gqa)
{
    validate_sdpa_input(query, key, value, attn_mask);

    static auto compatible_impl = at_npu::native::env::CheckCompatibleImpl();
    bool force_math = false;
    if (compatible_impl) {
        auto& ctx = at::globalContext();
        force_math = ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP();
        if (query.scalar_type() == at::kFloat &&
            (query.size(-1) % 4 != 0 || key.size(-1) % 4 != 0 || value.size(-1) % 4 != 0)) {
            force_math = true;
        }
    }

    if (!force_math &&
        (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat) &&
        ((attn_mask.has_value() && attn_mask->dtype() == at::kBool) || !attn_mask.has_value()) &&
        query.dim() == BNSD_DIM && key.dim() == BNSD_DIM && value.dim() == BNSD_DIM &&
        query.size(1) <= N_LIMIT && query.size(3) <= D_LIMIT && key.size(1) <= N_LIMIT &&
        query.size(1) % key.size(1) == 0 && query.size(1) / key.size(1) > 0 &&
        c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
        auto processed_mask = preprocess_attn_mask_for_npu(
            attn_mask, query, key);

        /* The implementation of the NPU FlashAttention fusion operator without grad constraints:
        1. The FA operator must be registered. GetSocVersion api is provisional, IsExistOp aclnn api will apply.
        2. The attn_mask supports only the bool data type.
        3. The shape [B, N1, S1, D] of the query is suppported, where N1 <= N_LIMIT,
            D <= D_LIMIT and dim == BNSD_DIM.
        4. For GQA, the key shape is [B, N2, S2, D], where N2 <= N_LIMIT, and N1 is a positive integer
            multiple of N2. */
        c10::optional<at::Tensor> atten_mask = convert_boolean_attn_mask(query, processed_mask, is_causal);
        int64_t head_num = query.size(1);
        c10::string_view input_layout = "BNSD";
        auto input_scale = calculate_scale(query, scale);
        double keep_prob = 1 - dropout_p;
        int64_t next_tockens = is_causal ? 0 : TOKEN_MAX;
        int64_t sparse_mode = is_causal ? LEFT_UP_CAUSAL : 0;
        c10::optional<at::Tensor> nulltensor = c10::nullopt;
        c10::OptionalArrayRef<c10::SymInt> nullsymlen = c10::nullopt;
        int64_t inner_precise = calculate_inner_precise_for_fa(query, attn_mask, is_causal);
        if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
            auto outputv3 =
                at_npu::native::custom_ops::npu_fusion_attention_v3(query, key, value, head_num, input_layout, nulltensor,
                                                                    nulltensor, atten_mask, input_scale.as_float_unchecked(),
                                                                    keep_prob, TOKEN_MAX, next_tockens, inner_precise, nullsymlen,
                                                                    nulltensor, nulltensor, sparse_mode, true, false);
            return std::get<0>(outputv3);
        }
        c10::OptionalIntArrayRef nulllen = c10::nullopt;
        TORCH_NPU_WARN_ONCE("CAUTION: On Ascend950, aten::scaled_dot_product_attention currently calls "
            "npu_fusion_attention. In graph mode, the seed, offset and numels int parameters may be optimized, "
            "which can cause results to differ from eager mode when the random seed is set. "
            "This issue will be fixed in later version.");
        auto output =
            at_npu::native::custom_ops::npu_fusion_attention(query, key, value, head_num, input_layout, nulltensor,
                                                             nulltensor, atten_mask, input_scale.as_float_unchecked(),
                                                             keep_prob, TOKEN_MAX, next_tockens, inner_precise, nulllen,
                                                             nulllen, nulllen, sparse_mode, true, false);
        return std::get<0>(output);
    } else if (!force_math &&
               (!query.requires_grad() && !key.requires_grad() && !value.requires_grad()) &&
                (query.dim() == BNSD_DIM && key.dim() == BNSD_DIM && value.dim() == BNSD_DIM) &&
                (query.size(0) != 0 && query.size(1) != 0 && query.size(2) != 0 && query.size(3) != 0) &&
                (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16) && query.size(3) % 16 ==0 &&
                ((attn_mask.has_value() && attn_mask->dtype() == at::kBool && attn_mask->size(-2) == query.size(2) &&
                attn_mask->size(-1) == key.size(2)) || !attn_mask.has_value()) &&
                (query.size(0) == key.size(0) && query.size(3) == key.size(3) && key.size(0) == value.size(0) &&
                key.size(1) == value.size(1) && key.size(2) == value.size(2) && key.size(3) == value.size(3)) &&
                (query.size(0) <= B_LIMIT && query.size(1) <= FIA_N_LIMIT && query.size(3) <= D_LIMIT && key.size(1) <= FIA_N_LIMIT) &&
                (key.size(1) != 0 && query.size(1) % key.size(1) == 0 && query.size(1) / key.size(1) <= 64) &&
                c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
                /* The implementation of the NPU Fused Infer Attention operator constraints:
                1. It only supports the type of fp16 or bf16, and dim = BNSD_DIM
                2. The shape [B, N1, S1, D] of the query is suppported, where B <= B_LIMIT, N1 <= FIA_N_LIMIT,
                    D <= D_LIMIT and must be a positive integer multiple of 16.
                3. The shape of the key and value should be the same, and dim = BNSD_DIM. For GQA, the key shape is [B, N2, S2, D],
                    where 0 < N2 <= FIA_N_LIMIT, N1 is a positive integer multiple of N2 and N1/N2 <= 64.
                4. The attn_mask supports only the bool data type, and shpae[-2] = S1, shpae[-1] = S2.
                5. It only supports SocVersion after Ascend910B1. */
                int64_t inner_precise = 0;
                if (query.size(2) == 1) {
                    is_causal = false;
                } else {
                    if (!is_causal) {
                        inner_precise = 2;
                    }
                }
                auto processed_mask_fia = preprocess_attn_mask_for_npu(
                    attn_mask, query, key);
                c10::optional<at::Tensor> atten_mask = convert_boolean_attn_mask(query, processed_mask_fia, is_causal);
                int64_t head_num = query.size(1);
                int64_t head_num_kv = key.size(1);
                c10::string_view input_layout = "BNSD";
                auto input_scale = calculate_scale(query, scale);
                int64_t next_tockens = is_causal ? 0 : TOKEN_MAX;
                int64_t sparse_mode = is_causal ? LEFT_UP_CAUSAL : 0;
                c10::optional<at::Tensor> nulltensor = c10::nullopt;
                at::OptionalSymIntArrayRef nulllen = c10::nullopt;
                auto output =
                    at_npu::native::custom_ops::npu_fused_infer_attention_score(query, key, value, nulltensor, atten_mask, nulllen, nulllen, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulltensor, nulltensor, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulltensor, nulltensor, nulltensor,
                                                                                nulltensor, nulltensor, nulltensor, nulllen, nulltensor, nulltensor, nulltensor, head_num, input_scale.as_float_unchecked(),
                                                                                TOKEN_MAX, next_tockens, input_layout, head_num_kv, sparse_mode, inner_precise, 0, 0, 0, 0, false);

                return std::get<0>(output);
    }
    c10::optional<at::Tensor> atten_mask_math = convert_boolean_attn_mask_math(attn_mask, query.dtype());
    auto output = at::_scaled_dot_product_attention_math(query, key, value, atten_mask_math, dropout_p, is_causal,
                                                         c10::nullopt, scale, enable_gqa);
    return std::get<0>(output);
}
#endif

}  // namespace op_api
