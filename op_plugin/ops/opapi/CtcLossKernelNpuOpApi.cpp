// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

std::tuple<at::Tensor, at::Tensor> _ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    bool zero_infinity)
{
    DO_COMPATIBILITY(aclnnCtcLoss, acl_op::_ctc_loss(log_probs, targets, input_lengths_list, target_lengths_list, blank,
                                                     zero_infinity));

    int64_t max_length = 0;
    for (auto& i : target_lengths_list) {
        if (i > max_length) {
            max_length = i;
        }
    }

    // calculate the output size
    auto outputSizes = op_infer::ctc_loss_npu_output_size(log_probs, max_length);

    // construct the output tensor of the NPU
    at::Tensor neg_log_likelihood = npu_preparation::apply_tensor_without_format(log_probs, std::get<0>(outputSizes));
    at::Tensor log_alpha = npu_preparation::apply_tensor_without_format(log_probs, std::get<1>(outputSizes));

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnCtcLoss, log_probs, targets, input_lengths_list, target_lengths_list,
                 blank, zero_infinity, neg_log_likelihood, log_alpha);

    return std::tuple<at::Tensor, at::Tensor>(neg_log_likelihood, log_alpha);
}

}
