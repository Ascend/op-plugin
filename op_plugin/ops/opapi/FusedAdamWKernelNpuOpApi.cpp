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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
void _fused_adamw_(
    at::TensorList self,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grade_scale,
    const c10::optional<at::Tensor>& found_inf)
{
    bool is_same_size = (self.size() == grads.size() &&
                       self.size() == exp_avgs.size() &&
                       self.size() == exp_avg_sqs.size() &&
                       self.size() == state_steps.size() &&
                       (max_exp_avg_sqs.size() == 0 ||
                       self.size() == max_exp_avg_sqs.size()));
    if (!is_same_size) {
        TORCH_CHECK(false, "the size of tensor list should be same.");
    }

    float lr_cast = static_cast<float>(lr);
    float beta1_cast = static_cast<float>(beta1);
    float beta2_cast = static_cast<float>(beta2);
    float weight_decay_cast = static_cast<float>(weight_decay);
    float eps_cast = static_cast<float>(eps);

    for (size_t i = 0; i < self.size(); i++) {
        auto step = state_steps[i].sub(1);
        // max_exp_avg_sqs is optional when amsgrad is false
        if (max_exp_avg_sqs.size() == 0) {
            c10::optional<at::Tensor> null_max_exp;
            EXEC_NPU_CMD(aclnnApplyAdamWV2, self[i], exp_avgs[i], exp_avg_sqs[i], null_max_exp, grads[i],
                step, lr_cast, beta1_cast, beta2_cast, weight_decay_cast, eps_cast, amsgrad, maximize);
        } else {
            EXEC_NPU_CMD(aclnnApplyAdamWV2, self[i], exp_avgs[i], exp_avg_sqs[i], max_exp_avg_sqs[i], grads[i],
                step, lr_cast, beta1_cast, beta2_cast, weight_decay_cast, eps_cast, amsgrad, maximize);
        }
    }
}
}  // namespace op_api
