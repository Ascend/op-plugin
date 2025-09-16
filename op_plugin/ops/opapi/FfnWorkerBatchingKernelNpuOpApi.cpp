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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

    static const int64_t NUM_TWO = 2;
    static const int64_t NUM_THREE = 3;
    static const int64_t MAX_OUT_SHAPE_DIM = 4;

    using npu_preparation = at_npu::native::OpPreparation;
    using npu_utils = at_npu::native::NpuUtils;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_ffn_worker_batching(
        const at::Tensor& schedule_context,
        int64_t expert_num,
        at::IntArrayRef max_out_shape,
        int64_t token_dtype,
        int64_t need_schedule,
        int64_t layer_num)
    {
        TORCH_CHECK(expert_num > 0, "expert_num should be a positive number");

        TORCH_CHECK(!max_out_shape.empty(), "max_out_shape is required");
        TORCH_CHECK(max_out_shape.size() == MAX_OUT_SHAPE_DIM, "max_out_shape must have 4 elements");
        TORCH_CHECK(max_out_shape[0] > 0, "first element of max_out_shape should be a positive number");
        TORCH_CHECK(max_out_shape[1] > 0, "second element of max_out_shape should be a positive number");
        TORCH_CHECK(max_out_shape[NUM_TWO] > 0, "third element of max_out_shape should be a positive number");
        TORCH_CHECK(max_out_shape[NUM_THREE] > 0, "fourth element of max_out_shape should be a positive number");

        TORCH_CHECK(token_dtype == 0 || token_dtype == 1 || token_dtype == NUM_TWO,
                    "token_dtype should be 0, 1 or 2, but got ", token_dtype);

        TORCH_CHECK(need_schedule == 0 || need_schedule == 1,
                    "need_schedule should be 0 or 1 , but got ", need_schedule);

        auto hs_dtype = at::kHalf;
        if (token_dtype == 1) {
            hs_dtype = at::kBFloat16;
        }
        if (token_dtype == NUM_TWO) {
            hs_dtype = at::kChar;
        }

        int Y = max_out_shape[0] * max_out_shape[1] * max_out_shape[NUM_TWO];
        int H = max_out_shape[NUM_THREE];

        at::Tensor y = npu_preparation::apply_tensor_without_format({Y, H}, schedule_context.options().dtype(hs_dtype));
        at::Tensor group_list = npu_preparation::apply_tensor_without_format({expert_num, NUM_TWO}, schedule_context.options().dtype(at::kLong));
        at::Tensor session_ids = npu_preparation::apply_tensor_without_format({Y}, schedule_context.options().dtype(at::kInt));
        at::Tensor micro_batch_ids = npu_preparation::apply_tensor_without_format({Y}, schedule_context.options().dtype(at::kInt));
        at::Tensor token_ids = npu_preparation::apply_tensor_without_format({Y}, schedule_context.options().dtype(at::kInt));
        at::Tensor expert_offsets = npu_preparation::apply_tensor_without_format({Y}, schedule_context.options().dtype(at::kInt));
        at::Tensor dynamic_scale = npu_preparation::apply_tensor_without_format({Y}, schedule_context.options().dtype(at::kFloat));
        at::Tensor actual_token_num = npu_preparation::apply_tensor_without_format({1}, schedule_context.options().dtype(at::kLong));

        EXEC_NPU_CMD(aclnnFfnWorkerBatching, schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num,
            y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num);

        return std::tie(y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num);
    }
}
