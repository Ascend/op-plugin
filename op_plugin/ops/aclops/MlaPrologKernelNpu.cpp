// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
namespace {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> npu_mla_prolog_nocheck(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    const at::Tensor& cache_index, const at::Tensor& kv_cache, const at::Tensor& kr_cache,
    const at::Tensor& dequant_scale_x, const at::Tensor& dequant_scale_w_dq,
    const at::Tensor& dequant_scale_w_uq_qr, const at::Tensor& dequant_scale_w_dkv_kr,
    const at::Tensor& quant_scale_ckv, const at::Tensor& quant_scale_ckr,
    const at::Tensor& smooth_scales_cq, double rmsnorm_epsilon_cq,
    double rmsnorm_epsilon_ckv, c10::string_view cache_mode, at::Tensor& query,
    at::Tensor& query_rope, at::Tensor& kv_cache_out, at::Tensor& kr_cache_out)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("MlaProlog")
       .Input(token_x, "token_x")
       .Input(weight_dq, "weight_dq")
       .Input(weight_uq_qr, "weight_uq_qr")
       .Input(weight_uk, "weight_uk")
       .Input(weight_dkv_kr, "weight_dkv_kr")
       .Input(rmsnorm_gamma_cq, "rmsnorm_gamma_cq")
       .Input(rmsnorm_gamma_ckv, "rmsnorm_gamma_ckv")
       .Input(rope_sin, "rope_sin")
       .Input(rope_cos, "rope_cos")
       .Input(cache_index, "cache_index")
       .Input(kv_cache, "kv_cache")
       .Input(kr_cache, "kr_cache");

    if (dequant_scale_x.defined()) {
        cmd.Input(dequant_scale_x, "dequant_scale_x");
    } else {
        cmd.Input();
    }

    if (dequant_scale_w_dq.defined()) {
        cmd.Input(dequant_scale_w_dq, "dequant_scale_w_dq");
    } else {
        cmd.Input();
    }

    if (dequant_scale_w_uq_qr.defined()) {
        cmd.Input(dequant_scale_w_uq_qr, "dequant_scale_w_uq_qr");
    } else {
        cmd.Input();
    }

    if (dequant_scale_w_dkv_kr.defined()) {
        cmd.Input(dequant_scale_w_dkv_kr, "dequant_scale_w_dkv_kr");
    } else {
        cmd.Input();
    }

    if (quant_scale_ckv.defined()) {
        cmd.Input(quant_scale_ckv, "quant_scale_ckv");
    } else {
        cmd.Input();
    }

    if (quant_scale_ckr.defined()) {
        cmd.Input(quant_scale_ckr, "quant_scale_ckr");
    } else {
        cmd.Input();
    }

    if (smooth_scales_cq.defined()) {
        cmd.Input(smooth_scales_cq, "smooth_scales_cq");
    } else {
        cmd.Input();
    }

    cmd.Output(query, "query")
       .Output(query_rope, "query_rope")
       .Output(kv_cache_out, "kv_cache_out")
       .Output(kr_cache_out, "kr_cache_out")
       .Attr("rmsnorm_epsilon_cq", static_cast<float>(rmsnorm_epsilon_cq))
       .Attr("rmsnorm_epsilon_ckv", static_cast<float>(rmsnorm_epsilon_ckv));

    cmd.Attr("cache_mode", (string)cache_mode).Run();

    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(query, query_rope, kv_cache_out, kr_cache_out);
}
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    const at::Tensor& cache_index, const at::Tensor& kv_cache, const at::Tensor& kr_cache,
    const c10::optional<at::Tensor>& dequant_scale_x_opt, const c10::optional<at::Tensor>& dequant_scale_w_dq_opt,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr_opt, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr_opt, const c10::optional<at::Tensor>& quant_scale_ckv_opt,
    const c10::optional<at::Tensor>& quant_scale_ckr_opt, const c10::optional<at::Tensor>& smooth_scales_cq_opt,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode)
{
    std::map<std::string, at::Tensor> mMap = { {"token_x", token_x}, {"weight_dq", weight_dq}, {"weight_uq_qr", weight_uq_qr}, {"weight_uk", weight_uk}, {"weight_dkv_kr", weight_dkv_kr}, {"rmsnorm_gamma_cq", rmsnorm_gamma_cq}, {"rmsnorm_gamma_ckv", rmsnorm_gamma_ckv}, {"rope_sin", rope_sin}, {"rope_cos", rope_cos}, {"cache_index", cache_index}, {"kv_cache", kv_cache}, {"kr_cache", kr_cache}};

    for (auto item : mMap) {
        TORCH_CHECK(item.second.numel() > 0, item.first.c_str(), " should not be null, but the actual value is null", OPS_ERROR(ErrCode::PARAM));
    }

    // construct the output tensor
    auto token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == 2 || token_x_dim == 3, "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim, OPS_ERROR(ErrCode::PARAM));

    auto weight_uk_dim = weight_uk.dim();
    TORCH_CHECK(weight_uk_dim == 3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim, OPS_ERROR(ErrCode::PARAM));

    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == 2 || rope_sin_dim == 3, "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));

    at::Tensor query;
    at::Tensor query_rope;

    if (token_x_dim == 3) {
        query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(2)}, token_x.options().dtype(token_x.dtype()));
    } else {
        query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), rope_sin.size(1)}, token_x.options().dtype(token_x.dtype()));
    }
    at::Tensor kv_cache_out = npu_preparation::apply_tensor_without_format(kv_cache);
    at::Tensor kr_cache_out = npu_preparation::apply_tensor_without_format(kr_cache);

    // optional inputs tensor
    const at::Tensor& dequant_scale_x = c10::value_or_else(dequant_scale_x_opt, [] { return at::Tensor(); });
    const at::Tensor& dequant_scale_w_dq = c10::value_or_else(dequant_scale_w_dq_opt, [] { return at::Tensor(); });
    const at::Tensor& dequant_scale_w_uq_qr = c10::value_or_else(dequant_scale_w_uq_qr_opt, [] { return at::Tensor(); });
    const at::Tensor& dequant_scale_w_dkv_kr = c10::value_or_else(dequant_scale_w_dkv_kr_opt, [] { return at::Tensor(); });
    const at::Tensor& quant_scale_ckv = c10::value_or_else(quant_scale_ckv_opt, [] { return at::Tensor(); });
    const at::Tensor& quant_scale_ckr = c10::value_or_else(quant_scale_ckr_opt, [] { return at::Tensor(); });
    const at::Tensor& smooth_scales_cq = c10::value_or_else(smooth_scales_cq_opt, [] { return at::Tensor(); });

    // check contiguous
    if (!npu_utils::check_match(&query) || !npu_utils::check_match(&query_rope) ||
        !npu_utils::check_match(&kv_cache_out) || !npu_utils::check_match(&kr_cache_out)) {
        // 若输出非连续，创建连续tensor(contig_tensor)，接收ACLOP算子的输出。再将contig_tensor拷贝到原始输出。
        at::Tensor contiguous_query = !npu_utils::check_match(&query) ? npu_utils::format_contiguous(query) : query;
        at::Tensor contiguous_query_rope = !npu_utils::check_match(&query_rope) ? npu_utils::format_contiguous(query_rope) : query_rope;
        at::Tensor contiguous_kv_cache_out = !npu_utils::check_match(&kv_cache_out) ?
            npu_utils::format_contiguous(kv_cache_out) : kv_cache_out;
        at::Tensor contiguous_kr_cache_out = !npu_utils::check_match(&kr_cache_out) ?
            npu_utils::format_contiguous(kr_cache_out) : kr_cache_out;
        npu_mla_prolog_nocheck(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache,
            dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr,
            smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, query, query_rope, kv_cache_out, kr_cache_out);
        npu_utils::format_fresh_view(query, contiguous_query);
        npu_utils::format_fresh_view(query_rope, contiguous_query_rope);
        npu_utils::format_fresh_view(kv_cache_out, contiguous_kv_cache_out);
        npu_utils::format_fresh_view(kr_cache_out, contiguous_kr_cache_out);
    } else {
        // 若输出连续，直接调用ACLOP算子。
        npu_mla_prolog_nocheck(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache,
            dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr,
            smooth_scales_cq, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
            cache_mode, query, query_rope, kv_cache_out, kr_cache_out);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, kv_cache_out, kr_cache_out);
}
} // namespace acl_op
