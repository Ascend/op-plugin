// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    constexpr int64_t TOKEN_FEATURE_DIM_IDX = 3;
    constexpr int64_t AVAILABLE_VERSION_NUM = 2;
    constexpr int64_t MAX_DIM = 4;
    constexpr int64_t V1_MODE_IDX = 0;
    constexpr int64_t V2_MODE_IDX = 1;
    constexpr int64_t DK_SIZES[AVAILABLE_VERSION_NUM] = {64, 192};
    constexpr int64_t DV_SIZES[AVAILABLE_VERSION_NUM] = {512, 128};

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_kv_rmsnorm_rope_cache(
        const at::Tensor &kv,
        const at::Tensor &gamma,
        const at::Tensor &cos,
        const at::Tensor &sin,
        const at::Tensor &index,
        const at::Tensor &k_cache,
        const at::Tensor &ckv_cache,
        const c10::optional<at::Tensor> &k_rope_scale,
        const c10::optional<at::Tensor> &c_kv_scale,
        const c10::optional<at::Tensor> &k_rope_offset,
        const c10::optional<at::Tensor> &c_kv_offset,
        const c10::optional<at::Tensor> &v,
        double epsilon,
        c10::string_view cache_mode,
        bool is_output_kv)
    {
        TORCH_CHECK((kv.dim() == MAX_DIM), "4D tensor expected for input kv" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((gamma.dim() == 1), "1D tensor expected for input gamma" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((cos.dim() == MAX_DIM), "4D tensor expected for input cos" + OPS_ERROR(ErrCode::PARAM));

        const at::Tensor &v_tsr_opt = v.value_or(at::Tensor());
        const int64_t vDim = v_tsr_opt.dim();
        static const bool is_kv_rnrc_V2_available = check_aclnn_kernel_available("aclnnKvRmsNormRopeCacheV2");
        c10::SmallVector<int64_t, SIZE> k_rope_shape = {0, 0, 0, 0};
        c10::SmallVector<int64_t, SIZE> c_kv_shape = {0, 0, 0, 0};
        for (int64_t i = 0; i < TOKEN_FEATURE_DIM_IDX; ++i) {
            k_rope_shape[i] = kv.size(i);
            c_kv_shape[i] = kv.size(i);
        }

        bool exec_v2_flag = is_kv_rnrc_V2_available && vDim == MAX_DIM && v_tsr_opt.size(TOKEN_FEATURE_DIM_IDX) == DV_SIZES[V2_MODE_IDX];
        if (exec_v2_flag) {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = DK_SIZES[V2_MODE_IDX];
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = DV_SIZES[V2_MODE_IDX];
        } else {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = cos.size(TOKEN_FEATURE_DIM_IDX);
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = gamma.size(0);
        }
        char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
        at::Tensor k_rope = npu_preparation::apply_tensor_without_format(k_rope_shape, kv.options());
        at::Tensor c_kv = npu_preparation::apply_tensor_without_format(c_kv_shape, kv.options());

        if (exec_v2_flag) {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCacheV2, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v_tsr_opt, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        }
        return std::tie(k_cache, ckv_cache, k_rope, c_kv);
    }

    std::tuple<at::Tensor, at::Tensor> npu_kv_rmsnorm_rope_cache_v2(
        const at::Tensor &kv,
        const at::Tensor &gamma,
        const at::Tensor &cos,
        const at::Tensor &sin,
        const at::Tensor &index,
        at::Tensor &k_cache,
        at::Tensor &ckv_cache,
        const c10::optional<at::Tensor> &k_rope_scale,
        const c10::optional<at::Tensor> &c_kv_scale,
        const c10::optional<at::Tensor> &k_rope_offset,
        const c10::optional<at::Tensor> &c_kv_offset,
        const c10::optional<at::Tensor> &v,
        double epsilon,
        c10::string_view cache_mode,
        bool is_output_kv)
    {
        TORCH_CHECK((kv.dim() == MAX_DIM), "4D tensor expected for input kv" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((gamma.dim() == 1), "1D tensor expected for input gamma" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((cos.dim() == MAX_DIM), "4D tensor expected for input cos" + OPS_ERROR(ErrCode::PARAM));

        const at::Tensor &v_tsr_opt = v.value_or(at::Tensor());
        const int64_t vDim = v_tsr_opt.dim();
        static const bool is_kv_rnrc_V2_available = check_aclnn_kernel_available("aclnnKvRmsNormRopeCacheV2");
        c10::SmallVector<int64_t, SIZE> k_rope_shape = {0, 0, 0, 0};
        c10::SmallVector<int64_t, SIZE> c_kv_shape = {0, 0, 0, 0};
        for (int64_t i = 0; i < TOKEN_FEATURE_DIM_IDX; ++i) {
            k_rope_shape[i] = kv.size(i);
            c_kv_shape[i] = kv.size(i);
        }

        bool exec_v2_flag = is_kv_rnrc_V2_available && vDim == MAX_DIM && v_tsr_opt.size(TOKEN_FEATURE_DIM_IDX) == DV_SIZES[V2_MODE_IDX];
        if (exec_v2_flag) {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = DK_SIZES[V2_MODE_IDX];
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = DV_SIZES[V2_MODE_IDX];
        } else {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = cos.size(TOKEN_FEATURE_DIM_IDX);
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = gamma.size(0);
        }
        char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
        at::Tensor k_rope = npu_preparation::apply_tensor_without_format(k_rope_shape, kv.options());
        at::Tensor c_kv = npu_preparation::apply_tensor_without_format(c_kv_shape, kv.options());

        if (exec_v2_flag) {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCacheV2, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v_tsr_opt, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        }
        return std::tie(k_rope, c_kv);
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_kv_rmsnorm_rope_cache_v2_functional(
        const at::Tensor &kv,
        const at::Tensor &gamma,
        const at::Tensor &cos,
        const at::Tensor &sin,
        const at::Tensor &index,
        const at::Tensor &k_cache,
        const at::Tensor &ckv_cache,
        const c10::optional<at::Tensor> &k_rope_scale,
        const c10::optional<at::Tensor> &c_kv_scale,
        const c10::optional<at::Tensor> &k_rope_offset,
        const c10::optional<at::Tensor> &c_kv_offset,
        const c10::optional<at::Tensor> &v,
        double epsilon,
        c10::string_view cache_mode,
        bool is_output_kv)
    {
        TORCH_CHECK((kv.dim() == MAX_DIM), "4D tensor expected for input kv" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((gamma.dim() == 1), "1D tensor expected for input gamma" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((cos.dim() == MAX_DIM), "4D tensor expected for input cos" + OPS_ERROR(ErrCode::PARAM));

        const at::Tensor &v_tsr_opt = v.value_or(at::Tensor());
        const int64_t vDim = v_tsr_opt.dim();
        static const bool is_kv_rnrc_V2_available = check_aclnn_kernel_available("aclnnKvRmsNormRopeCacheV2");
        c10::SmallVector<int64_t, SIZE> k_rope_shape = {0, 0, 0, 0};
        c10::SmallVector<int64_t, SIZE> c_kv_shape = {0, 0, 0, 0};
        for (int64_t i = 0; i < TOKEN_FEATURE_DIM_IDX; ++i) {
            k_rope_shape[i] = kv.size(i);
            c_kv_shape[i] = kv.size(i);
        }

        bool exec_v2_flag = is_kv_rnrc_V2_available && vDim == MAX_DIM && v_tsr_opt.size(TOKEN_FEATURE_DIM_IDX) == DV_SIZES[V2_MODE_IDX];
        if (exec_v2_flag) {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = DK_SIZES[V2_MODE_IDX];
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = DV_SIZES[V2_MODE_IDX];
        } else {
            k_rope_shape[TOKEN_FEATURE_DIM_IDX] = cos.size(TOKEN_FEATURE_DIM_IDX);
            c_kv_shape[TOKEN_FEATURE_DIM_IDX] = gamma.size(0);
        }

        char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
        at::Tensor k_rope = npu_preparation::apply_tensor_without_format(k_rope_shape, kv.options());
        at::Tensor c_kv = npu_preparation::apply_tensor_without_format(c_kv_shape, kv.options());
        at::Tensor k_cache_inplace = k_cache.clone();
        at::Tensor ckv_cache_inplace = ckv_cache.clone();

        if (exec_v2_flag) {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCacheV2, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v_tsr_opt, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        } else {
            EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                         k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode_ptr,
                                         is_output_kv, k_rope, c_kv);
        }
        return std::tie(k_rope, c_kv, k_cache_inplace, ckv_cache_inplace);
    }
    
}
