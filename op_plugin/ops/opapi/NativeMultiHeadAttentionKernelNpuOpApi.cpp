// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ops/_native_multi_head_attention_native.h>

#include <cmath>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
static const int64_t TOKEN_MAX = 2147483647;

// Convert the merged mask (as produced by `torch.nn.functional.multi_head_attention_forward::merge_masks`)
// into the `atten_mask` accepted by `npu_fusion_attention`.
//
// Mask semantics: both the ATen `_native_multi_head_attention` mask and `npu_fusion_attention`
// use True == mask out, False == keep, so NO logical inversion is applied here. (Note the
// contrast with ScaledDotProductAttentionKernelNpuOpApi.cpp, which calls `at::logical_not` on
// its input: the ATen `scaled_dot_product_attention` bool mask uses the opposite convention,
// True == keep, and the inversion adapts it to the fused op.)
static c10::optional<at::Tensor> prepare_mha_attn_mask(
    const c10::optional<at::Tensor>& mask,
    const c10::optional<int64_t>& mask_type,
    int64_t B,
    int64_t N,
    int64_t S)
{
    if (!mask.has_value()) {
        return c10::nullopt;
    }
    at::Tensor atten_mask = mask.value();
    // The fusion guard only dispatches bool masks here; additive float masks (0 / -inf
    // weights) are rejected because a silent `.to(kBool)` cast would collapse every
    // non-zero weight to True and silently corrupt the attention result.
    TORCH_CHECK(atten_mask.scalar_type() == at::kBool,
        "The fused attention path only supports bool masks, but got mask dtype: ",
        atten_mask.scalar_type(), OPS_ERROR(ErrCode::NOT_SUPPORT));
    if (!mask_type.has_value()) {
        // Defensive fall through: the fusion guard only enters with a valid mask_type.
        return atten_mask;
    }
    switch (mask_type.value()) {
        case 1:
            // key_padding_mask: [B, S] -> [B, 1, 1, S] -> [B, N, S, S].
            // NOTE: npu_fusion_attention on CANN 9.1.0 rejects broadcast layouts such as
            // [B, 1, 1, S] (aclnnFlashAttentionScore tiling failure), so the mask must be
            // materialized to the dense [B, N, S, S] shape here. This costs an N-fold
            // mask memory copy, which is unavoidable until the aclnn accepts broadcast
            // layouts.
            if (atten_mask.dim() == 2) {
                atten_mask = atten_mask.contiguous().view({B, 1, 1, S}).expand({B, N, S, S}).contiguous();
            }
            break;
        case 0:
            // src_mask: [S, S] broadcast over batch and head; the aclnn requires a dense
            // 4D mask (see the mask_type == 1 note), so materialize the expanded view.
            if (atten_mask.dim() == 2) {
                atten_mask = atten_mask.expand({B, N, S, S}).contiguous();
            }
            break;
        case 2:
            // default_mask: expected to be [B, N, S, S] already.
            TORCH_CHECK(atten_mask.dim() == 4,
                "Expected a 4D mask for mask_type == 2, but got dim: ", atten_mask.dim(),
                OPS_ERROR(ErrCode::VALUE));
            break;
        default:
            // Unknown layouts are passed through untouched.
            break;
    }
    return atten_mask;
}
#endif

std::tuple<at::Tensor, at::Tensor> _native_multi_head_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    const c10::optional<at::Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const c10::optional<int64_t> mask_type)
{
    TORCH_CHECK(num_head > 0,
        "num_head must be positive, but got: ", num_head, OPS_ERROR(ErrCode::VALUE));
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
    const auto B = query.size(0);
    const auto S = query.size(1);
    const bool is_self_attn = query.is_same(key) && key.is_same(value);
    const bool supported_dtype = query.scalar_type() == at::kFloat ||
        query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16;
    // Only bool masks take the fused path; additive float masks keep their weight
    // semantics only in the composite reference implementation below.
    const bool mask_fusable = !mask.has_value() ||
        (mask_type.has_value() && mask.value().scalar_type() == at::kBool);
    static const bool fusion_attention_available =
        check_aclnn_kernel_available("aclnnFusionAttention");

    // Boundary guard: only fuse plain self-attention without attention weights.
    if (fusion_attention_available && !need_weights && is_self_attn && query.numel() != 0 &&
        embed_dim % num_head == 0 && supported_dtype && mask_fusable) {
        const auto head_dim = embed_dim / num_head;
        // Packed QKV projection: query is [B, S, D], qkv_weight is [3 * D, D].
        // q is kept un-scaled here; the scaling is applied inside npu_fusion_attention via `scale`.
        auto qkv = at::linear(query, qkv_weight, qkv_bias);
        auto q_proj = qkv.slice(2, 0, embed_dim);
        auto k_proj = qkv.slice(2, embed_dim, 2 * embed_dim);
        auto v_proj = qkv.slice(2, 2 * embed_dim, 3 * embed_dim);

        // Reshape to BNSD ([B, N, S, head_dim]) for the fused attention op.
        auto q = q_proj.view({B, S, num_head, head_dim}).transpose(1, 2).contiguous();
        auto k = k_proj.view({B, S, num_head, head_dim}).transpose(1, 2).contiguous();
        auto v = v_proj.view({B, S, num_head, head_dim}).transpose(1, 2).contiguous();

        c10::optional<at::Tensor> atten_mask = prepare_mha_attn_mask(mask, mask_type, B, num_head, S);
        c10::optional<at::Tensor> nulltensor = c10::nullopt;
        c10::OptionalIntArrayRef nulllen = c10::nullopt;
        std::string_view input_layout = "BNSD";
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        double keep_prob = 1.0;
        // Mirror the SDPA accuracy knob: use high precision softmax when a mask is applied on multi-token sequences.
        int64_t inner_precise = (mask.has_value() && S > 1) ? 2 : 0;
        int64_t sparse_mode = 0;  // defaultMask
        int64_t head_num = num_head;

        auto output = at_npu::native::custom_ops::npu_fusion_attention(
            q, k, v, head_num, input_layout, nulltensor, nulltensor, atten_mask, scale, keep_prob,
            TOKEN_MAX, TOKEN_MAX, inner_precise, nulllen, nulllen, nulllen, sparse_mode, true, false);

        // attention_out is [B, N, S, head_dim] -> [B, S, D] -> out projection.
        auto attn_out = std::get<0>(output).transpose(1, 2).contiguous().view({B, S, embed_dim});
        auto out_proj = at::linear(attn_out, proj_weight, proj_bias);
        return std::make_tuple(out_proj, at::Tensor());
    }
#endif
    return at::native::native_multi_head_attention_cpu(
        query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
        proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
}

}  // namespace op_api
