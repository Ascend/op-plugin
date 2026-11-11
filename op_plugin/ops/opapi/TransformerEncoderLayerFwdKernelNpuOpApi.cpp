// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <ATen/ops/_transformer_encoder_layer_fwd_native.h>

namespace op_api {

// Single Transformer encoder layer forward. Delegates to the ATen native
// implementation, which composes the layer-norm, self-attention and FFN sub-blocks
// out of basic tensor ops plus `_native_multi_head_attention`; the latter is fused on
// NPU, so the whole layer runs on NPU through the dispatcher.
at::Tensor _transformer_encoder_layer_fwd(
    const at::Tensor& src,
    const int64_t embed_dim,
    const int64_t num_heads,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    const bool use_gelu,
    const bool norm_first,
    const double layer_norm_eps,
    const at::Tensor& layer_norm_weight_1,
    const at::Tensor& layer_norm_bias_1,
    const at::Tensor& layer_norm_weight_2,
    const at::Tensor& layer_norm_bias_2,
    const at::Tensor& ffn_weight_1,
    const at::Tensor& ffn_bias_1,
    const at::Tensor& ffn_weight_2,
    const at::Tensor& ffn_bias_2,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<int64_t> mask_type)
{
    return at::native::transformer_encoder_layer_forward(
        src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias,
        use_gelu, norm_first, layer_norm_eps, layer_norm_weight_1, layer_norm_bias_1,
        layer_norm_weight_2, layer_norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2,
        ffn_bias_2, mask, mask_type);
}

}  // namespace op_api
