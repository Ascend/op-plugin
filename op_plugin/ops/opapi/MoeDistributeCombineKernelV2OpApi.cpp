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
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    const int DIM_TWO = 2;

at::Tensor npu_moe_distribute_combine_v2(const at::Tensor &expand_x, const at::Tensor &expert_ids,
                                         const at::Tensor &assist_info_for_combine,
                                         const at::Tensor &ep_send_counts, const at::Tensor &expert_scales,
                                         c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                         int64_t moe_expert_num,
                                         const c10::optional<at::Tensor> &tp_send_counts,
                                         const c10::optional<at::Tensor> &x_active_mask,
                                         const c10::optional<at::Tensor> &expand_scales,
                                         const c10::optional<at::Tensor> &shared_expert_x,
                                         c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
                                         int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                         int64_t global_bs, int64_t comm_quant_mode)
{
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf) || (expand_x.scalar_type() == at::kInt),
                "dtype of expand_x should be BFloat16, Float16 or Int, but got " + std::string(c10::toString(expand_x.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;

    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    std::string group_tp_str = std::string(group_tp);
    char *group_tp_ptr = const_cast<char *>(group_tp_str.c_str());
    at::Tensor output;
    if (expand_x.scalar_type() != at::kInt) {
        output = npu_preparation::apply_tensor_without_format({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    } else {
        output = npu_preparation::apply_tensor_without_format({bs, h}, expert_ids.options().dtype(at::kBFloat16));
    }
    
    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t out_dtype = 0;
    int64_t group_list_type = 0;
    std::string comm_log = "0";
    char *comm_log_ptr = const_cast<char *>(comm_log.c_str());
    EXEC_NPU_CMD(aclnnMoeDistributeCombineV2, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, tp_send_counts, x_active_mask,
                 nulltensor, nulltensor, nulltensor, expand_scales, shared_expert_x, group_ep_ptr, ep_world_size, ep_rank_id,
                 moe_expert_num, group_tp_ptr, tp_world_size, tp_rank_id,
                 expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real, out_dtype, comm_quant_mode, group_list_type,
                 comm_log_ptr, output);
    return output;
}
}
