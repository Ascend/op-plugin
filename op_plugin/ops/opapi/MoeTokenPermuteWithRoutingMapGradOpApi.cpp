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

::std::tuple<at::Tensor, at::Tensor> npu_moe_token_permute_with_routing_map_grad_symint(const at::Tensor & permuted_token_out_grad, const c10::optional<at::Tensor> & probs_grad, const at::Tensor & sorted_indices, const at::Tensor & routing_map, c10::SymInt experts_num, c10::SymInt tokens_num, bool drop_and_pad)
{
    auto token_num_int = tokens_num.expect_int();
    auto experts_num_int = experts_num.expect_int();
    auto output_size_0 = {token_num_int, permuted_token_out_grad.size(1)};
    auto output_size_1 = {token_num_int, experts_num.expect_int()};
    auto output_dtype_0 = permuted_token_out_grad.scalar_type();
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0, permuted_token_out_grad.options().dtype(output_dtype_0));
    at::Tensor out2;
    if (probs_grad.has_value() && probs_grad.value().defined()) {
        out2 = npu_preparation::apply_tensor_without_format(output_size_1, probs_grad.value().options().dtype(probs_grad.value().scalar_type()));
    }
    EXEC_NPU_CMD(aclnnMoeTokenPermuteWithRoutingMapGrad, permuted_token_out_grad, probs_grad, sorted_indices, routing_map, experts_num_int, token_num_int, drop_and_pad, out1, out2);
    return std::make_tuple(std::move(out1), std::move(out2));
}
}
