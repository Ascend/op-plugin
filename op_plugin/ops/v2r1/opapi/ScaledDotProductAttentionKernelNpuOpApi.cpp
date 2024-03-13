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

namespace op_api {
const static int64_t ATTENMASK_LIMIT = 2048;
const static int64_t N_LIMIT = 2048;
const static int64_t D_LIMIT = 512;
const static int64_t BNSD_DIM = 4;
const static int64_t TOKEN_MAX = 2147483647;
const static int64_t LEFT_UP_CAUSAL = 2;
using npu_preparation = at_npu::native::OpPreparation;

inline void validate_sdpa_input(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale)
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
        at::Tensor atten_mask_shape =
            npu_preparation::apply_tensor_without_format({ATTENMASK_LIMIT, ATTENMASK_LIMIT},
                                                         c10::dtype(c10::ScalarType::Bool));
        at::Tensor atten_mask_comp = op_api::logical_not(atten_mask_shape);
        auto new_attn_mask = op_api::triu(atten_mask_comp, 1);
        return new_attn_mask;
    }
    const at::Tensor &atten_mask_in = attn_mask.value_or(at::Tensor());
    at::Tensor atten_mask = op_api::logical_not(atten_mask_in);
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

at::Tensor scaled_dot_product_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale)
{
    validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale);
    if (query.requires_grad() && key.requires_grad() && value.requires_grad() &&
        (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16) &&
        ((attn_mask.has_value() && attn_mask->dtype() == at::kBool) || !attn_mask.has_value()) &&
        query.dim() == BNSD_DIM && key.dim() == BNSD_DIM && value.dim() == BNSD_DIM &&
        query.size(1) <= N_LIMIT && query.size(3) <= D_LIMIT && key.size(1) <= N_LIMIT &&
        query.size(1) % key.size(1) == 0 && query.size(1) / key.size(1) > 0 &&
        c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
        /* The implementation of the NPU FlashAttention fusion operator constraints:
           1. The attn_mask supports only the bool data type.
           2. The shape [B, N1, S1, D] of the query is suppported, where N1 <= N_LIMIT,
              D <= D_LIMIT and dim == BNSD_DIM.
           3. For GQA, the key shape is [B, N2, S2, D], where N2 <= N_LIMIT, and N1 is a positive integer
              multiple of N2.
           4. It only supports SocVersion after Ascend910B1. */
        c10::optional<at::Tensor> atten_mask = convert_boolean_attn_mask(attn_mask, is_causal);
        int64_t head_num = query.size(1);
        c10::string_view input_layout = "BNSD";
        auto input_scale = calculate_scale(query, scale);
        double keep_prob = 1 - dropout_p;
        int64_t next_tockens = is_causal ? 0 : TOKEN_MAX;
        int64_t sparse_mode = is_causal ? LEFT_UP_CAUSAL : 0;
        c10::optional<at::Tensor> nulltensor = c10::nullopt;
        c10::OptionalIntArrayRef nulllen = c10::nullopt;
        auto output =
            at_npu::native::custom_ops::npu_fusion_attention(query, key, value, head_num, input_layout, nulltensor,
                                                             nulltensor, atten_mask, input_scale.as_float_unchecked(),
                                                             keep_prob, TOKEN_MAX, next_tockens, 0, nulllen,
                                                             nulllen, nulllen, sparse_mode, true, false);
        return std::get<0>(output);
    }
    c10::optional<at::Tensor> atten_mask_math = convert_boolean_attn_mask_math(attn_mask, query.dtype());
    auto output = at::_scaled_dot_product_attention_math(query, key, value, atten_mask_math, dropout_p, is_causal,
                                                         c10::nullopt, scale);
    return std::get<0>(output);
}
}  // namespace op_api
