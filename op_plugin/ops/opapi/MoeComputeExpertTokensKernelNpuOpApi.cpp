// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
at::Tensor npu_moe_compute_expert_tokens(const at::Tensor &sorted_expert_for_source_row, const int64_t num_expert)
{
    c10::SmallVector<int64_t, SIZE> output_size = {num_expert};
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(sorted_expert_for_source_row,
                                                                                   output_size);
    EXEC_NPU_CMD(aclnnMoeComputeExpertTokens, sorted_expert_for_source_row, num_expert, result);

    return result;
}

}
