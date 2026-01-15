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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
void prepare_before_quant(
    at::IntArrayRef qkv_size, at::IntArrayRef head_nums, const at::Tensor& qkv, bool is_output_qkv,
    at::Tensor& q_out_before_quant, at::Tensor& k_out_before_quant, at::Tensor& v_out_before_quant)
{
    TORCH_CHECK(
        (qkv_size.size() == DIM_FOUR),
        "npu_qkv_rms_norm_rope_cache: the size of qkv_size must be 4." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (head_nums.size() == DIM_THREE),
        "npu_qkv_rms_norm_rope_cache: the size of head_nums must be 3." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (qkv.dim() == DIM_TWO),
        "npu_qkv_rms_norm_rope_cache: 2D tensor expected for input qkv" + OPS_ERROR(ErrCode::PARAM));
    int64_t Nq = head_nums[0];
    int64_t Nk = head_nums[1];
    int64_t Nv = head_nums[2];
    int64_t D = qkv_size[3];

    c10::SmallVector<int64_t, DIM_TWO> q_out_before_quant_shape = {qkv.size(DIM_ZERO), Nq * D};
    c10::SmallVector<int64_t, DIM_TWO> k_out_before_quant_shape = {qkv.size(DIM_ZERO), Nk * D};
    c10::SmallVector<int64_t, DIM_TWO> v_out_before_quant_shape = {qkv.size(DIM_ZERO), Nv * D};
    if (is_output_qkv) {
        q_out_before_quant = npu_preparation::apply_tensor_without_format(q_out_before_quant_shape, qkv.options());
        k_out_before_quant = npu_preparation::apply_tensor_without_format(k_out_before_quant_shape, qkv.options());
        v_out_before_quant = npu_preparation::apply_tensor_without_format(v_out_before_quant_shape, qkv.options());
    } else {
        q_out_before_quant = at::Tensor();
        k_out_before_quant = at::Tensor();
        v_out_before_quant = at::Tensor();
    }
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_qkv_rms_norm_rope_cache(
    const at::Tensor& qkv, const at::Tensor& q_gamma, const at::Tensor& k_gamma, const at::Tensor& cos,
    const at::Tensor& sin, const at::Tensor& index, at::Tensor& q_out, at::Tensor& k_cache, at::Tensor& v_cache,
    at::IntArrayRef qkv_size, at::IntArrayRef head_nums, const c10::optional<at::Tensor>& k_scale,
    const c10::optional<at::Tensor>& v_scale, const c10::optional<at::Tensor>& k_offset,
    const c10::optional<at::Tensor>& v_offset, double epsilon, c10::string_view cache_mode, bool is_output_qkv)
{
    at::Tensor q_out_before_quant;
    at::Tensor k_out_before_quant;
    at::Tensor v_out_before_quant;
    prepare_before_quant(
        qkv_size, head_nums, qkv, is_output_qkv, q_out_before_quant, k_out_before_quant, v_out_before_quant);
    char* cache_mode_ptr = const_cast<char*>(cache_mode.data());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnQkvRmsNormRopeCache, qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, k_scale, v_scale,
        k_offset, v_offset, qkv_size, head_nums, epsilon, cache_mode_ptr, q_out_before_quant, k_out_before_quant,
        v_out_before_quant);

    return std::tie(q_out_before_quant, k_out_before_quant, v_out_before_quant);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_qkv_rms_norm_rope_cache_functional(
    const at::Tensor& qkv, const at::Tensor& q_gamma, const at::Tensor& k_gamma, const at::Tensor& cos,
    const at::Tensor& sin, const at::Tensor& index, const at::Tensor& q_out, const at::Tensor& k_cache,
    const at::Tensor& v_cache, at::IntArrayRef qkv_size, at::IntArrayRef head_nums,
    const c10::optional<at::Tensor>& k_scale, const c10::optional<at::Tensor>& v_scale,
    const c10::optional<at::Tensor>& k_offset, const c10::optional<at::Tensor>& v_offset, double epsilon,
    c10::string_view cache_mode, bool is_output_qkv)
{
    at::Tensor q_out_before_quant;
    at::Tensor k_out_before_quant;
    at::Tensor v_out_before_quant;
    prepare_before_quant(
        qkv_size, head_nums, qkv, is_output_qkv, q_out_before_quant, k_out_before_quant, v_out_before_quant);
    char* cache_mode_ptr = const_cast<char*>(cache_mode.data());

    at::Tensor q_out_inplace = q_out.clone();
    at::Tensor k_cache_inplace = k_cache.clone();
    at::Tensor v_cache_inplace = v_cache.clone();
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnQkvRmsNormRopeCache, qkv, q_gamma, k_gamma, cos, sin, index, q_out_inplace, k_cache_inplace,
        v_cache_inplace, k_scale, v_scale, k_offset, v_offset, qkv_size, head_nums, epsilon, cache_mode_ptr,
        q_out_before_quant, k_out_before_quant, v_out_before_quant);

    return std::tie(
        q_out_before_quant, k_out_before_quant, v_out_before_quant, q_out_inplace, k_cache_inplace, v_cache_inplace);
}
} // namespace op_api