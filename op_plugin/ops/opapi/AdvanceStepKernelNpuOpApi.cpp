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
using npu_preparation = at_npu::native::OpPreparation;

void npu_advance_step_flashattn(at::Tensor &input_tokens, const at::Tensor &sampled_token_ids,
    at::Tensor &input_positions, at::Tensor &seq_lens, at::Tensor &slot_mapping, const at::Tensor &block_tables,
    int64_t num_seqs, int64_t num_queries, int64_t block_size, const c10::optional<at::Tensor> &spec_token,
    const c10::optional<at::Tensor> &accepted_num)
{
    if (spec_token.has_value() || accepted_num.has_value()) {
        EXEC_NPU_CMD(aclnnAdvanceStepV2,
            input_tokens,
            sampled_token_ids,
            input_positions,
            seq_lens,
            slot_mapping,
            block_tables,
            spec_token,
            accepted_num,
            num_seqs,
            num_queries,
            block_size);
    } else {
        EXEC_NPU_CMD(aclnnAdvanceStep,
            input_tokens,
            sampled_token_ids,
            input_positions,
            seq_lens,
            slot_mapping,
            block_tables,
            num_seqs,
            num_queries,
            block_size);
    }
}

}  // namespace op_api