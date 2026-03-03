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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _mrope_v1(
    const at::Tensor &positions,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos_sin_cache,
    at::IntArrayRef mrope_section,
    int64_t head_size,
    bool is_neox_style,
    at::Tensor &query_out,
    at::Tensor &key_out)
{
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnRopeWithSinCosCache, positions, query, key, cos_sin_cache,
        mrope_section, head_size, is_neox_style, query_out, key_out);
    return std::tie(query_out, key_out);
}

std::tuple<at::Tensor, at::Tensor> npu_mrope(
    const at::Tensor &positions,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos_sin_cache,
    int64_t head_size,
    c10::OptionalIntArrayRef mrope_section,
    c10::optional<c10::string_view> rotary_mode,
    c10::optional<c10::string_view> cache_mode)
{
    at::IntArrayRef mrope_section_value = mrope_section.value_or(at::IntArrayRef{0, 0, 0});
    std::string rotary_mode_str = rotary_mode.has_value() ? std::string(rotary_mode.value()) : "half";
    std::string cache_mode_str = cache_mode.has_value() ? std::string(cache_mode.value()) : "default";
    
    bool is_neox_style = op_plugin::utils::is_neox_style(rotary_mode_str);
    int64_t cache_mode_value = op_plugin::utils::cache_mode_to_int(cache_mode_str);
    
    static const bool is_mrope_v2_available = check_aclnn_kernel_available("aclnnRopeWithSinCosCacheV2");
    
    TORCH_CHECK(
        is_mrope_v2_available || cache_mode_value == 0,
        "npu_mrope: cache_mode='", cache_mode_str,
        "' requires aclnnRopeWithSinCosCacheV2, but current environment does not support it. "
        "Please upgrade CANN or use cache_mode='default'.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    
    at::Tensor query_out = at::empty_like(query);
    at::Tensor key_out = at::empty_like(key);
    
    DO_COMPATIBILITY(aclnnRopeWithSinCosCacheV2,
        _mrope_v1(positions, query, key, cos_sin_cache, mrope_section_value,
            head_size, is_neox_style, query_out, key_out));

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnRopeWithSinCosCacheV2, positions, query, key, cos_sin_cache,
        mrope_section_value, head_size, is_neox_style, cache_mode_value, query_out, key_out);
    return std::tie(query_out, key_out);
}
}

