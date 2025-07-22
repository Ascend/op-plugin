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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_token_permute_with_routing_map(const at::Tensor & tokens, const at::Tensor & routing_map, const c10::optional<at::Tensor> & probs, c10::optional<int64_t> num_out_tokens, bool drop_and_pad)
{
    auto num_out_tokens_value = num_out_tokens.value_or(tokens.size(0));
    auto out_token = num_out_tokens_value / routing_map.size(drop_and_pad) * routing_map.size(drop_and_pad);
    auto output_size_0 = {out_token, tokens.size(1)};
    auto output_size_1 = {out_token};
    auto output_dtype_0 = tokens.scalar_type();
    auto output_dtype_1 = at::kInt;
    at::Tensor out2;
    if (probs.has_value() && probs.value().defined()) {
        out2 = npu_preparation::apply_tensor_without_format(output_size_1, tokens.options().dtype(probs.value().scalar_type()));
    }
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0, tokens.options().dtype(output_dtype_0));
    at::Tensor out3 = npu_preparation::apply_tensor_without_format(output_size_1, tokens.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeTokenPermuteWithRoutingMap, tokens, routing_map, probs, num_out_tokens_value, drop_and_pad, out1, out2, out3);
    return std::make_tuple(std::move(out1), std::move(out2), std::move(out3));
}
}
