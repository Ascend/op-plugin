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
    double epsilon,
    c10::string_view cache_mode,
    bool is_output_kv)
{
    TORCH_CHECK((kv.dim() == 4), "4D tensor expected for input kv" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((gamma.dim() == 1), "1D tensor expected for input gamma" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((cos.dim() == 4), "4D tensor expected for input cos" + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> k_rope_shape = {kv.size(0), kv.size(1), kv.size(2), cos.size(3)};
    c10::SmallVector<int64_t, SIZE> c_kv_shape = {kv.size(0), kv.size(1), kv.size(2), gamma.size(0)};

    char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
    at::Tensor k_rope = npu_preparation::apply_tensor_without_format(k_rope_shape, kv.options());
    at::Tensor c_kv = npu_preparation::apply_tensor_without_format(c_kv_shape, kv.options());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                 k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, epsilon, cache_mode_ptr,
                                 is_output_kv, k_rope, c_kv);
    return std::tie(k_cache, ckv_cache, k_rope, c_kv);
}

}
