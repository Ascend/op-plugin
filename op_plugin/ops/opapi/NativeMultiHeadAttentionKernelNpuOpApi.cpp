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

#include <ATen/ops/_native_multi_head_attention_native.h>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _native_multi_head_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    const c10::optional<at::Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const c10::optional<int64_t> mask_type)
{
    return at::native::native_multi_head_attention_cpu(
        query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
        proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
}

}  // namespace op_api
