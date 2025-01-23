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

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_rope_quant_kvcache(
        const at::Tensor &x,
        const at::Tensor &cos,
        const at::Tensor &sin,
        const at::Tensor &k_cache_ref,
        const at::Tensor &v_cache_ref,
        const at::Tensor &indices,
        const at::Tensor &scale_k,
        const at::Tensor &scale_v,
        at::IntArrayRef size_splits,
        const c10::optional<at::Tensor> &offset_k_optional,
        const c10::optional<at::Tensor> &offset_v_optional,
        int64_t quant_mode,
        c10::string_view input_layout,
        const bool kv_output,
        c10::string_view cache_mode)
    {
        TORCH_CHECK((size_splits[0] >= 0), "size_splits[0] should not less than 0" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((x.dim() == 3 || x.dim() == 2), "3D or 2D tensor expected for input x" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((v_cache_ref.dim() == 4), "4D tensor expected for input cache" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK((size_splits.size() == 3), "size_splits's size should be 3" + OPS_ERROR(ErrCode::PARAM));
        
        const int64_t b = x.size(0);
        const int64_t s = x.size(1);
        const int64_t kv_headdim = v_cache_ref.size(2);
        const int64_t d = v_cache_ref.size(3);
        const int64_t q_headdim = (d == 0) ? 0 : size_splits[0] / d;

        c10::SmallVector<int64_t, SIZE> q_shape = {b, q_headdim, d};
        if (x.dim() == 3) {
            q_shape = {b, s, q_headdim, d};
        }

        char *quant_mode_ptr = "static";
        if (quant_mode == 1) {
            quant_mode_ptr = "dynamic";
        }
        char *input_layout_ptr = const_cast<char *>(input_layout.data());
        char *cache_mode_ptr = const_cast<char *>(cache_mode.data());

        at::Tensor q_output = npu_preparation::apply_tensor_without_format(q_shape, cos.options());
        at::Tensor k_output;
        at::Tensor v_output;
        at::Tensor weight;
        at::Tensor scale;
        at::Tensor bias;

        if (kv_output) {
            c10::SmallVector<int64_t, SIZE> k_shape = {b, kv_headdim, d};
            c10::SmallVector<int64_t, SIZE> v_shape = {b, kv_headdim, d};
            if (x.dim() == 3) {
                k_shape = {b, s, kv_headdim, d};
                v_shape = {b, s, kv_headdim, d};
            }
            k_output = npu_preparation::apply_tensor_without_format(k_shape, cos.options());
            v_output = npu_preparation::apply_tensor_without_format(v_shape, cos.options());
        } else {
            k_output = npu_preparation::apply_tensor_without_format({0}, cos.options());
            v_output = npu_preparation::apply_tensor_without_format({0}, cos.options());
        }

        EXEC_NPU_CMD(aclnnDequantRopeQuantKvcache, x, cos, sin, k_cache_ref,
                     v_cache_ref, indices, scale_k, scale_v,
                     offset_k_optional, offset_v_optional, weight, scale, bias,
                     size_splits, quant_mode_ptr, input_layout_ptr, kv_output, cache_mode_ptr,
                     q_output, k_output, v_output);
        return std::tie(q_output, k_output, v_output, k_cache_ref, v_cache_ref);
    }
} // namespace op_api
