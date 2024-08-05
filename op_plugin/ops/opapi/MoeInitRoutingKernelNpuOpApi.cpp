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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_init_routing(const at::Tensor &x, const at::Tensor &row_idx, const at::Tensor &expert_idx, int64_t active_num)
{
    TORCH_NPU_WARN_ONCE("The oprator of MoeInitRouting will be removed from Pytorch and switch to AscendSpeed after 630.");
    TORCH_CHECK(x.dim() == DIM_TWO, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());
    TORCH_CHECK(expert_idx.dim() == DIM_TWO, "The expert_idx should be 2D");
    TORCH_CHECK(
        expert_idx.scalar_type() == at::kInt,
        "int32 tensor expected but got a tensor with dtype: ",
        expert_idx.scalar_type());
    TORCH_CHECK(row_idx.dim() == DIM_TWO, "The row_idx should be 2D");
    TORCH_CHECK(
        row_idx.scalar_type() == at::kInt,
        "int32 tensor expected but got a tensor with dtype: ",
        row_idx.scalar_type());
    TORCH_CHECK(active_num >= 0, "The active_num must be a non-negative number");

    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    auto row_idx_size = row_idx.sizes();
    TORCH_CHECK(x_size[0] == expert_idx_size[0], "Input rows shoud be same.");
    TORCH_CHECK(expert_idx_size == row_idx_size, "The shape of expert_idx and row_idx should be same.");

    int n = x_size[0];
    int cols = x_size[1];
    int k = expert_idx_size[1];
    active_num = n > active_num ? active_num : n;
    at::Tensor expanded_x = npu_preparation::apply_tensor_without_format(x, {active_num * k, cols});
    at::Tensor expanded_row_idx = npu_preparation::apply_tensor_without_format(row_idx, {n * k});
    at::Tensor expanded_expert_idx = npu_preparation::apply_tensor_without_format(expert_idx, {n * k});
    EXEC_NPU_CMD(aclnnMoeInitRouting, x, row_idx, expert_idx, active_num, expanded_x, expanded_row_idx, expanded_expert_idx);
    return std::tie(expanded_x, expanded_row_idx, expanded_expert_idx);
}

}
