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
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    const int DIM_TWO = 2;

tensor_list npu_moe_distribute_dispatch(const at::Tensor &x, const at::Tensor &expert_ids,
                                        c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                        int64_t moe_expert_num,
                                        const c10::optional<at::Tensor> &scales,
                                        const c10::optional<at::Tensor> &x_active_mask,
                                        const c10::optional<at::Tensor> &expert_scales,
                                        c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
                                        int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                        int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type)
{
    TORCH_CHECK((x.dim() == 2) && (expert_ids.dim() == 2), "The x and expert_ids should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((x.scalar_type() == at::kBFloat16 || (x.scalar_type() == at::kHalf)) && (expert_ids.scalar_type() == at::kInt),
                "dtype of x should be bfloat16 or half, dtype of expert_ids should be int.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((shared_expert_rank_num >= 0) && (shared_expert_rank_num < ep_world_size),
                "shared_expert_rank_num should be in [0, ep_world_size)", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((expert_token_nums_type == 0) || (expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.", OPS_ERROR(ErrCode::PARAM));
    auto x_size = x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t n = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_ids_size[1];

    // a2 expert_shard_type、shared_expert_rank_num 应为0
    bool shared_front = (expert_shard_type == 0) ? true : false;
    int64_t local_moe_expert_num = 0;
    int64_t global_bs_real = (global_bs == 0) ? (n * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
    if (shared_front) {
        if (ep_rank_id < shared_expert_rank_num) {
            local_moe_expert_num =  1;
            a = global_bs_real / shared_expert_rank_num;
        } else {
            local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
            a = global_bs_real * std::min(local_moe_expert_num, k);
        }
    } else {
        if (ep_rank_id >= ep_world_size - shared_expert_rank_num) {
            local_moe_expert_num = 1;
            a = global_bs_real / shared_expert_rank_num;
        } else {
            local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
            a = global_bs_real * std::min(local_moe_expert_num, k);
        }
    }
    if (tp_world_size == DIM_TWO) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size;
    } else {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num;
    }

    auto output_dtype = (!scales.has_value() && quant_mode == 0) ? x.scalar_type() : at::kChar;
    char *group_ep_ptr = const_cast<char *>(group_ep.data());
    std::string group_tp_str = std::string(group_tp);
    char *group_tp_ptr = const_cast<char *>(group_tp_str.c_str());
    at::Tensor expand_x {nullptr};
    at::Tensor dynamic_scales {nullptr};
    if (tp_world_size == 0) {
        expand_x = npu_preparation::apply_tensor_without_format({a, h}, x.options().dtype(output_dtype));
        dynamic_scales = npu_preparation::apply_tensor_without_format({a}, x.options().dtype(at::kFloat));
    } else {
        expand_x = npu_preparation::apply_tensor_without_format({a * tp_world_size, h}, x.options().dtype(output_dtype));
        dynamic_scales = npu_preparation::apply_tensor_without_format({a * tp_world_size}, x.options().dtype(at::kFloat));
    }
    
    at::Tensor expand_idx = npu_preparation::apply_tensor_without_format({n * k}, x.options().dtype(at::kInt));
    at::Tensor expert_token_nums = npu_preparation::apply_tensor_without_format({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor tp_recv_counts = npu_preparation::apply_tensor_without_format({tp_world_size}, x.options().dtype(at::kInt));

    // a2分层方案
    at::Tensor expand_scales = npu_preparation::apply_tensor_without_format({a}, x.options().dtype(at::kFloat));
    if (expert_scales.has_value() && expert_scales.value().defined()) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8); // 2: 2 buffer, 8 ranknum per server
        ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatch, x, expert_ids, scales, x_active_mask, expert_scales,
                 group_ep_ptr, ep_world_size, ep_rank_id,
                 moe_expert_num,
                 group_tp_ptr, tp_world_size, tp_rank_id,
                 expert_shard_type, shared_expert_num, shared_expert_rank_num,
                 quant_mode, global_bs_real, expert_token_nums_type, expand_x,
                 dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts,
                 tp_recv_counts, expand_scales);
    return std::tie(expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}
}
