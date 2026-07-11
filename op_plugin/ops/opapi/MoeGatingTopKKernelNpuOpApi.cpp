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
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_moe_gating_top_k(const at::Tensor &x, int64_t k, const c10::optional<at::Tensor> &bias_opt, const c10::optional<at::Tensor> &input_ids_opt, const c10::optional<at::Tensor> &tid2eid_opt, int64_t k_group, int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag, double routed_scaling_factor, double eps)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());

    auto x_size = x.sizes();
    auto rows = x_size[0];
    auto expert_num = x_size[1];
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    if (bias.defined()) {
        TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "The dtype of x and bias should be same");
        TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
        auto bias_size = bias.sizes();
        TORCH_CHECK(bias_size[0] == expert_num, "The bias first dim should be same as x second dim");
    }

    const at::Tensor &input_ids = c10::value_or_else(input_ids_opt, [] { return at::Tensor(); });
    const at::Tensor &tid2eid = c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });
    TORCH_CHECK(input_ids.defined() == tid2eid.defined(),
        "input_ids and tid2eid must be both provided or both omitted");

    if (input_ids.defined()) {
        TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
            "int32 or int64 tensor expected for input_ids but got a tensor with dtype: ",
            input_ids.scalar_type());
        TORCH_CHECK(input_ids.dim() == 1, "The input_ids should be 1D");
        auto input_ids_size = input_ids.sizes();
        TORCH_CHECK(input_ids_size[0] == rows, "The input_ids first dim should be same as x first dim");
    }

    if (tid2eid.defined()) {
        TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
            "int32 or int64 tensor expected for tid2eid but got a tensor with dtype: ",
            tid2eid.scalar_type());
        TORCH_CHECK(tid2eid.dim() == 2, "The tid2eid should be 2D");
        auto tid2eid_size = tid2eid.sizes();
        TORCH_CHECK(tid2eid_size[1] == k, "The tid2eid second dim should be same as k");
    }

    at::Tensor y = npu_preparation::apply_tensor_without_format(x, {rows, k});
    at::Tensor expert_idx = npu_preparation::apply_tensor_without_format({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = npu_preparation::apply_tensor_without_format({rows, expert_num}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMoeGatingTopKV2, x, bias, input_ids, tid2eid, k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps, y, expert_idx, out);
    return std::tie(y, expert_idx, out);
}

}
