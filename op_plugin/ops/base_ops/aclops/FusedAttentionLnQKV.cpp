// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(
    const at::Tensor &x, const at::Tensor &kernel_query, const at::Tensor &kernel_key, const at::Tensor &kernel_value,
    const at::Tensor &gamma, const at::Tensor &beta, const c10::optional<at::Tensor> &bias_query,
    const c10::optional<at::Tensor> &bias_key, const c10::optional<at::Tensor> &bias_value, int64_t seq_len,
    int64_t num_heads, double eps)
{
    TORCH_CHECK(seq_len != 0 || num_heads != 0, "seq_len and num_heads cannot be equal to 0."
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2 dimensions, but got ", x.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(0) % seq_len == 0 && x.size(1) % num_heads == 0,
        "In npu_fused_attention_layernorm_qkv_fwd, x.size(0) should be "
        "divisible by seq_len",
        "and x.size(1) should be divisible by num_heads."
        + OPS_ERROR(ErrCode::PARAM));
    const at::Tensor &bias_query_input = c10::value_or_else(bias_query, [] { return at::Tensor(); });
    const at::Tensor &bias_key_input = c10::value_or_else(bias_key, [] { return at::Tensor(); });
    const at::Tensor &bias_value_input = c10::value_or_else(bias_value, [] { return at::Tensor(); });

    c10::SmallVector<int64_t, SIZE> qkv_output_shape = {x.size(0) / seq_len, num_heads, seq_len, x.size(1) / num_heads};
    c10::SmallVector<int64_t, SIZE> mean_output_shape = {x.size(0)};
    at::Tensor norm = npu_preparation::apply_tensor(x);
    at::Tensor query_output = npu_preparation::apply_tensor(kernel_query, qkv_output_shape);
    at::Tensor key_output = npu_preparation::apply_tensor(kernel_key, qkv_output_shape);
    at::Tensor value_output = npu_preparation::apply_tensor(kernel_value, qkv_output_shape);
    at::Tensor mean = npu_preparation::apply_tensor_with_format(kernel_query, mean_output_shape, ACL_FORMAT_ND);
    at::Tensor variance = npu_preparation::apply_tensor_with_format(kernel_query, mean_output_shape, ACL_FORMAT_ND);

    at_npu::native::OpCommand cmd;
    cmd.Name("AttentionLnQKV")
        .Input(x)
        .Input(kernel_query)
        .Input(kernel_key)
        .Input(kernel_value)
        .Input(gamma)
        .Input(beta)
        .Input(bias_query_input)
        .Input(bias_key_input)
        .Input(bias_value_input)
        .Output(norm)
        .Output(query_output)
        .Output(key_output)
        .Output(value_output)
        .Output(mean)
        .Output(variance)
        .Attr("epsilon", static_cast<float>(eps))
        .Run();
    std::vector<at::Tensor> results = {norm, query_output, key_output, value_output, mean, variance};
    return results;
}
} // namespace acl_op
