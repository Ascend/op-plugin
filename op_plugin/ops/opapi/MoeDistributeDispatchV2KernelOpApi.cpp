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

tensor_list npu_moe_distribute_dispatch_v2(const at::Tensor &x, const at::Tensor &expert_ids,
                                           c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
                                           int64_t moe_expert_num,
                                           const c10::optional<at::Tensor> &scales,
                                           const c10::optional<at::Tensor> &x_active_mask,
                                           const c10::optional<at::Tensor> &expert_scales,
                                           c10::string_view group_tp, int64_t tp_world_size, int64_t tp_rank_id,
                                           int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                           int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type)
{
    TORCH_CHECK((x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "dtype of x should be BFloat16 or Half, but got " + std::string(c10::toString(x.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", ep_rank_id: ", ep_rank_id,
                ". " + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((shared_expert_rank_num >= 0) && (shared_expert_rank_num < ep_world_size),
                "shared_expert_rank_num should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". " + OPS_ERROR(ErrCode::PARAM));
    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    bool is_valid_shared = ((shared_expert_num > 0)
        && ((shared_expert_rank_num / shared_expert_num) > 0)
        && ((shared_expert_rank_num % shared_expert_num) == 0));
    TORCH_CHECK(is_shared_default || is_no_shared || is_valid_shared,
                "shared_expert_num and shared_expertrank_num have obvious value situations: "
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert num is 0, "
                "shared_expert_rank_num is 0; 3. shared_expert_num in (0, shared_expert_rank_num] and "
                "shared_expert_rank_num % shared_expert_num = 0. but the current input value is ",
                " shared_expert_num: ", shared_expert_num,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". " + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((expert_token_nums_type == 0) || (expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.", OPS_ERROR(ErrCode::PARAM));
    auto x_size = x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_ids_size[1];

    // a2 expert_shard_type、shared_expert_rank_num 应为0
    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 0;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
    if (shared_front) {
        if (ep_rank_id < shared_expert_rank_num) {
            local_moe_expert_num =  1;
            int64_t max_bs = global_bs_real / ep_world_size;  // 前面已有拦截，保证ep_world_size > 0
            int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;  // 前面已有拦截, 保证进入该分支时shared_expert_num > 0
            int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
            a = max_bs * max_shared_group_num;
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
    
    at::Tensor expert_token_nums = npu_preparation::apply_tensor_without_format({local_moe_expert_num}, x.options().dtype(at::kLong));
    at::Tensor ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    at::Tensor tp_recv_counts = npu_preparation::apply_tensor_without_format({tp_world_size}, x.options().dtype(at::kInt));
    at::Tensor assist_info_forcombine{nullptr};

    // a2分层方案
    at::Tensor expand_scales = npu_preparation::apply_tensor_without_format({a}, x.options().dtype(at::kFloat));
    if (expert_scales.has_value() && expert_scales.value().defined()) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8); // 2: 2 buffer, 8 ranknum per server
        ep_recv_counts = npu_preparation::apply_tensor_without_format({ep_recv_cnt_num}, x.options().dtype(at::kInt));
    }

    std::string comm_log = "0";
    char *comm_log_ptr = const_cast<char *>(comm_log.c_str());
    assist_info_forcombine = npu_preparation::apply_tensor_without_format({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2, x, expert_ids, scales, x_active_mask, expert_scales,
                 group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num,
                 group_tp_ptr, tp_world_size, tp_rank_id,
                 expert_shard_type, shared_expert_num, shared_expert_rank_num,
                 quant_mode, global_bs_real, expert_token_nums_type, comm_log_ptr, expand_x,
                 dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts,
                 tp_recv_counts, expand_scales);
    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}
}
