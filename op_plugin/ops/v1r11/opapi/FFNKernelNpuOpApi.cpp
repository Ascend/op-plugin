// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
const static int MIN_DIM = 2;
const static int X_MAX_DIM = 8;
const static int EXPERT_WEIGHT_DIM = 3;
const static int NO_EXPERT_WEIGHT_DIM = 2;
const static int MAX_EXPERT_NUM = 256;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_ffn(const at::Tensor &x, const at::Tensor &weight1, const at::Tensor &weight2,
    c10::string_view activation, c10::optional<at::IntArrayRef> expert_tokens, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &bias2, c10::optional<int64_t> inner_precise)
{
    auto weight1_dim_num = weight1.dim();
    auto weight2_dim_num = weight2.dim();
    auto x_dim_num = x.dim();
    TORCH_CHECK(x_dim_num >= MIN_DIM && x_dim_num <= X_MAX_DIM, "x shape dims should be 2~8, but it is ", x_dim_num);
    auto x_k_dim = x.size(x.dim() - 1);
    auto wight1_k_dim = weight1.size(weight1.dim() - 2);
    TORCH_CHECK(x_k_dim == wight1_k_dim, "The k of x and weight should be equal. but x_k_dim is ",
        x_k_dim, ", wight1_k_dim is ", wight1_k_dim);

    char *activation_ptr = const_cast<char *>(activation.data());
    const at::Tensor &bias1_real = bias1.value_or(at::Tensor());
    const at::Tensor &bias2_real = bias2.value_or(at::Tensor());
    const at::Tensor &scale = at::Tensor();
    const at::Tensor &offset = at::Tensor();
    const at::Tensor &deq_scale1 = at::Tensor();
    const at::Tensor &deq_scale2 = at::Tensor();
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[x.dim() - 1] = weight2.size(weight2.dim() - 1);
    at::Tensor result = npu_preparation::apply_tensor_without_format(x, output_size);
    int64_t inner_precise_val = inner_precise.has_value() ? inner_precise.value() : 0;
    if (expert_tokens.has_value()) {
        auto expert_tokens_real = expert_tokens.value();
        auto token_size = expert_tokens_real.size();
        TORCH_CHECK(token_size <= MAX_EXPERT_NUM, "expert_tokens should be smaller than 256, but it is ", token_size);
        TORCH_CHECK(weight1_dim_num == EXPERT_WEIGHT_DIM && weight2_dim_num == EXPERT_WEIGHT_DIM,
            "The dimension of weight(has expert_tokens) should be 3, but weight1_dim_num is ",
            weight1_dim_num, ", weight2_dim_num is ", weight2_dim_num);
        EXEC_NPU_CMD(aclnnFFN, x, weight1, weight2, expert_tokens_real, bias1_real, bias2_real,
            scale, offset, deq_scale1, deq_scale2, activation_ptr, inner_precise_val, result);
    } else {
        auto expert_tokens_empty = at::Tensor();
        TORCH_CHECK(weight1_dim_num == NO_EXPERT_WEIGHT_DIM && weight2_dim_num == NO_EXPERT_WEIGHT_DIM,
            "The dimension of weight(no expert_tokens) should be 2, but weight1_dim_num is ",
            weight1_dim_num, ", weight2_dim_num is ", weight2_dim_num);
        EXEC_NPU_CMD(aclnnFFN, x, weight1, weight2, expert_tokens_empty, bias1_real, bias2_real,
            scale, offset, deq_scale1, deq_scale2, activation_ptr, inner_precise_val, result);
    }

    return result;
}
}  // namespace op_api