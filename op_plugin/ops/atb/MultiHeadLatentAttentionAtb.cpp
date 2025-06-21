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
#include "op_plugin/utils/custom_functions/atb/Utils.h"

namespace atb {

namespace {
std::unordered_map<c10::string_view, int> mask_type_map = {
    {"undefined", 0},
    {"mask_type_spec", 1},
    {"mask_type_free", 2}
};

std::unordered_map<c10::string_view, int> calc_type_map = {
    {"calc_type_undefined", 0},
    {"calc_type_spec", 1},
};

std::unordered_map<c10::string_view, int> cache_mode_map = {
    {"krope_ctkv", 1},
    {"int8_nzcache", 2},
    {"nzcache", 3}
};

std::tuple<int, int, int> get_mla_mode(c10::optional<c10::string_view> mask_type_opt,
                                       c10::optional<c10::string_view> calc_type_opt,
                                       c10::optional<c10::string_view> cache_mode_opt)
{
    int mask_type = atb::utils::get_op_mode(
        mask_type_map, mask_type_opt, "undefined", "mask_type");
    int calc_type = atb::utils::get_op_mode(
        calc_type_map, calc_type_opt, "calc_type_undefined", "calc_type");
    int cache_mode = atb::utils::get_op_mode(
        cache_mode_map, cache_mode_opt, "krope_ctkv", "cache_mode");
    return std::make_tuple(mask_type, calc_type, cache_mode);
}
}

at::Tensor npu_multi_head_latent_attention(const at::Tensor & q_nope, const at::Tensor & q_rope, const at::Tensor & ctkv, const at::Tensor & k_rope, const at::Tensor & block_tables,
    c10::SymIntArrayRef context_lens, int64_t head_num, double qk_scale, int64_t kv_headnum, const c10::optional<at::Tensor> & mask, c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor> & qk_descale, const c10::optional<at::Tensor> & pv_descale, c10::optional<c10::string_view> mask_type_opt, c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q_nope));
    at::Tensor output = at::empty(q_nope.sizes(), q_rope.options());
    at::Tensor lse;
    float qkScale_float = static_cast<float>(qk_scale);
    auto mode = get_mla_mode(mask_type_opt, calc_type_opt, cache_mode_opt);
    int mask_type = std::get<0>(mode);
    int calc_type = std::get<1>(mode);
    int cache_mode = std::get<2>(mode);
    at::Tensor context_lens_tensor = at::tensor(c10::asIntArrayRefUnchecked(context_lens), at::kInt);
    at::Tensor qseqlen_tensor = qseqlen.has_value()? at::tensor(c10::asIntArrayRefUnchecked(qseqlen.value()), at::kInt): at::Tensor();
    EXEC_ATB_CMD(AtbMLA, q_nope, q_rope, ctkv, k_rope, block_tables, context_lens_tensor, mask, qseqlen_tensor, qk_descale, pv_descale, head_num, qkScale_float, kv_headnum, mask_type, calc_type, cache_mode, output, lse);
    return output;
}

at::Tensor& npu_multi_head_latent_attention_out(const at::Tensor & q_nope, const at::Tensor & q_rope, const at::Tensor & ctkv, const at::Tensor & k_rope, const at::Tensor & block_tables,
    c10::SymIntArrayRef context_lens,  int64_t head_num, double qk_scale, int64_t kv_headnum, const c10::optional<at::Tensor> & mask, c10::OptionalArrayRef<c10::SymInt> qseqlen,
    const c10::optional<at::Tensor> & qk_descale, const c10::optional<at::Tensor> & pv_descale, c10::optional<c10::string_view> mask_type_opt, c10::optional<c10::string_view> calc_type_opt,
    c10::optional<c10::string_view> cache_mode_opt, at::Tensor & output)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q_nope));
    float qkScale_float = static_cast<float>(qk_scale);
    at::Tensor lse;
    auto mode = get_mla_mode(mask_type_opt, calc_type_opt, cache_mode_opt);
    int mask_type = std::get<0>(mode);
    int calc_type = std::get<1>(mode);
    int cache_mode = std::get<2>(mode);
    at::Tensor context_lens_tensor = at::tensor(c10::asIntArrayRefUnchecked(context_lens), at::kInt);
    at::Tensor qseqlen_tensor = qseqlen.has_value()? at::tensor(c10::asIntArrayRefUnchecked(qseqlen.value()), at::kInt): at::Tensor();
    EXEC_ATB_CMD(AtbMLA, q_nope, q_rope, ctkv, k_rope, block_tables, context_lens_tensor, mask, qseqlen_tensor, qk_descale, pv_descale, head_num, qkScale_float, kv_headnum, mask_type, calc_type, cache_mode, output, lse);
    return output;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_multi_head_latent_attention(Tensor q_nope, Tensor q_rope, Tensor ctkv, Tensor k_rope, Tensor block_tables, SymInt[] context_lens, int q_headnum, float qk_scale, int kv_headnum,"
    " *, Tensor? mask=None, SymInt[]? qseqlen=None, Tensor? qk_descale=None, Tensor? pv_descale=None, str? mask_type=None, str? calc_type=None, str? cache_mode=None) -> Tensor");
    m.def("npu_multi_head_latent_attention.out(Tensor q_nope, Tensor q_rope, Tensor ctkv, Tensor k_rope, Tensor block_tables, SymInt[] context_lens, int q_headnum, float qk_scale, int kv_headnum,"
    " *, Tensor? mask=None, SymInt[]? qseqlen=None, Tensor? qk_descale=None, Tensor? pv_descale=None, str? mask_type=None, str? calc_type=None, str? cache_mode=None, Tensor(a!) output) -> Tensor(a!)");
}
}
namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_multi_head_latent_attention", TORCH_FN(atb::npu_multi_head_latent_attention));
    m.impl("npu_multi_head_latent_attention.out", TORCH_FN(atb::npu_multi_head_latent_attention_out));
}
}

} // namespace atb
