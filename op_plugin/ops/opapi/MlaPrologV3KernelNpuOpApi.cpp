// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const int DIM_2 = 2;
const int DIM_3 = 3;
const int MODE_1 = 1;
const int MODE_2 = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
    const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr, const c10::optional<at::Tensor>& quant_scale_ckv,
    const c10::optional<at::Tensor>& quant_scale_ckr, const c10::optional<at::Tensor>& smooth_scales_cq,
    const c10::optional<at::Tensor>& actual_seq_len, const c10::optional<at::Tensor>& k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    // construct the output tensor
    auto token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3, "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim, OPS_ERROR(ErrCode::PARAM));

    auto weight_uk_dim = weight_uk.dim();
    TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim, OPS_ERROR(ErrCode::PARAM));

    auto rope_sin_dim = rope_sin.dim();

    at::Tensor query;
    at::Tensor query_rope;
    at::Tensor dequant_scale_q_nope {nullptr};
    at::Tensor query_norm {nullptr};
    at::Tensor dequant_scale_q_norm {nullptr};

    if (token_x_dim == DIM_3) {
        TORCH_CHECK(rope_sin_dim == DIM_3, "rope_sin dim num should be 3, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));
        if (weight_quant_mode == MODE_2 && kv_cache_quant_mode == MODE_1) { // weight_quant_mode=2且kv_cache_quant_mode=1时为全量化kv量化场景
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(2)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_quant_mode == MODE_1 || weight_quant_mode == MODE_2) { // weight_quant_mode=1 半量化场景,weight_quant_mode=2 全量化场景
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), 1}, at::kFloat);
            }
        }
    } else {
        TORCH_CHECK(rope_sin_dim == DIM_2, "rope_sin dim num should be 2, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));
        if (weight_quant_mode == MODE_2 && kv_cache_quant_mode == MODE_1) { // weight_quant_mode=2且kv_cache_quant_mode=1时为全量化kv量化场景
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), rope_sin.size(1)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_quant_mode == MODE_1 || weight_quant_mode == MODE_2) { // weight_quant_mode=1 半量化场景,weight_quant_mode=2 全量化场景
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0), 1}, at::kFloat);
            }
        }
    }

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());

    EXEC_NPU_CMD(aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq,
        dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
        k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query, query_rope,
        dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3_functional(
    const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
    const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
    const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
    const at::Tensor& kv_cache, const at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
    const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
    const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr, const c10::optional<at::Tensor>& quant_scale_ckv,
    const c10::optional<at::Tensor>& quant_scale_ckr, const c10::optional<at::Tensor>& smooth_scales_cq,
    const c10::optional<at::Tensor>& actual_seq_len, const c10::optional<at::Tensor>& k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
    int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    // construct the output tensor
    auto token_x_dim = token_x.dim();
    TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3, "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim, OPS_ERROR(ErrCode::PARAM));

    auto weight_uk_dim = weight_uk.dim();
    TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim, OPS_ERROR(ErrCode::PARAM));

    auto rope_sin_dim = rope_sin.dim();
    
    at::Tensor query;
    at::Tensor query_rope;
    at::Tensor dequant_scale_q_nope {nullptr};
    at::Tensor query_norm {nullptr};
    at::Tensor dequant_scale_q_norm {nullptr};

    if (token_x_dim == DIM_3) {
        TORCH_CHECK(rope_sin_dim == DIM_3, "rope_sin dim num should be 3, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));
        if (weight_quant_mode == MODE_2 && kv_cache_quant_mode == MODE_1) { // weight_quant_mode=2且kv_cache_quant_mode=1时为全量化kv量化场景
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(2)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), token_x.size(1), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_quant_mode == MODE_1 || weight_quant_mode == MODE_2) { // weight_quant_mode=1 半量化场景,weight_quant_mode=2 全量化场景
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0) * token_x.size(1), 1}, at::kFloat);
            }
        }
    } else {
        TORCH_CHECK(rope_sin_dim == DIM_2, "rope_sin dim num should be 2, but the actual value is ", rope_sin_dim, OPS_ERROR(ErrCode::PARAM));
        if (weight_quant_mode == MODE_2 && kv_cache_quant_mode == MODE_1) { // weight_quant_mode=2且kv_cache_quant_mode=1时为全量化kv量化场景
            // kvcache量化
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(token_x.dtype()));
            dequant_scale_q_nope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), 1}, at::kFloat);
        } else {
            query = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), weight_uk.size(2)}, token_x.options().dtype(rope_sin.dtype()));
        }
        query_rope = npu_preparation::apply_tensor_without_format({token_x.size(0), weight_uk.size(0), rope_sin.size(1)}, at::kBFloat16);
        if (query_norm_flag) {
            query_norm =  npu_preparation::apply_tensor_without_format({token_x.size(0), weight_dq.size(1)}, token_x.options().dtype(weight_uq_qr.dtype()));
            if (weight_quant_mode == MODE_1 || weight_quant_mode == MODE_2) { // weight_quant_mode=1 半量化场景,weight_quant_mode=2 全量化场景
                dequant_scale_q_norm = npu_preparation::apply_tensor_without_format({token_x.size(0), 1}, at::kFloat);
            }
        }
    }

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
    at::Tensor kv_cache_inplace = kv_cache.clone();
    at::Tensor kr_cache_inplace = kr_cache.clone();

    EXEC_NPU_CMD(aclnnMlaPrologV3WeightNz, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_inplace, kr_cache_inplace, cache_index, dequant_scale_x, dequant_scale_w_dq,
        dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
        k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query, query_rope,
        dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_inplace, kr_cache_inplace);
}

} // namespace op_api
