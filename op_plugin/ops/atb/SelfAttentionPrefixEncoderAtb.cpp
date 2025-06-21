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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

namespace atb {

namespace {
std::unordered_map<c10::string_view, int> mask_type_map = {
    {"mask_type_norm_compress", 3},
    {"mask_type_alibi_compress", 4},
    {"mask_type_alibi_compress_sort", 5},
    {"mask_type_causal_mask", 9}
};
}

at::Tensor npu_self_attention_prefix_encoder(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale, int64_t kv_headnum, const c10::optional<at::Tensor> & mask, const c10::optional<at::Tensor> & slopes, c10::optional<c10::string_view> mask_type_opt)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    at::Tensor output = at::empty(query.sizes(), query.options());
    auto mask_type = atb::utils::get_op_mode(mask_type_map, mask_type_opt, "mask_type_causal_mask", "mask_type");
    float qkScale_float = static_cast<float>(qk_scale);
    at::Tensor kv_seqlen_tensor = at::tensor(c10::asIntArrayRefUnchecked(kv_seqlen), at::kInt);
    at::Tensor seqlen_tensor = at::tensor(c10::asIntArrayRefUnchecked(seqlen), at::kInt);
    EXEC_ATB_CMD(AtbSelfAttentionPrefixEncoder, query, key, value, block_tables, mask, seqlen_tensor, kv_seqlen_tensor, slopes, mask_type, q_headnum, kv_headnum, qkScale_float, output);
    return output;
}

at::Tensor& npu_self_attention_prefix_encoder_out(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & block_tables, c10::SymIntArrayRef seqlen,
    c10::SymIntArrayRef kv_seqlen, int64_t q_headnum, double qk_scale, int64_t kv_headnum, const c10::optional<at::Tensor> & mask, const c10::optional<at::Tensor> & slopes, c10::optional<c10::string_view> mask_type_opt, at::Tensor & output)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    auto mask_type = atb::utils::get_op_mode(mask_type_map, mask_type_opt, "mask_type_causal_mask", "mask_type");
    float qkScale_float = static_cast<float>(qk_scale);
    at::Tensor kv_seqlen_tensor = at::tensor(c10::asIntArrayRefUnchecked(kv_seqlen), at::kInt);
    at::Tensor seqlen_tensor = at::tensor(c10::asIntArrayRefUnchecked(seqlen), at::kInt);
    EXEC_ATB_CMD(AtbSelfAttentionPrefixEncoder, query, key, value, block_tables, mask, seqlen_tensor, kv_seqlen_tensor, slopes, mask_type, q_headnum, kv_headnum, qkScale_float, output);
    return output;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_self_attention_prefix_encoder(Tensor query, Tensor key, Tensor value, Tensor block_tables, SymInt[] seqlen, SymInt[] kv_seqlen, int q_headnum, float qk_scale, int kv_headnum,*, Tensor? mask=None, Tensor? slopes=None, str? mask_type=None) -> Tensor");
    m.def("npu_self_attention_prefix_encoder.out(Tensor query, Tensor key, Tensor value, Tensor block_tables, SymInt[] seqlen, SymInt[] kv_seqlen, int q_headnum, float qk_scale, int kv_headnum,*, Tensor? mask=None, Tensor? slopes=None, str? mask_type=None, Tensor(a!) output) -> Tensor(a!)");
}
}
namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_self_attention_prefix_encoder", TORCH_FN(atb::npu_self_attention_prefix_encoder));
    m.impl("npu_self_attention_prefix_encoder.out", TORCH_FN(atb::npu_self_attention_prefix_encoder_out));
}
}

} // namespace atb
