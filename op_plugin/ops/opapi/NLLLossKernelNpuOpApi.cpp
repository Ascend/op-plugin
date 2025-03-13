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

static std::tuple<at::Tensor&, at::Tensor&> nll_loss_forward_npu_nocheck(
    at::Tensor& result, at::Tensor& total_weight, const at::Tensor& self, const at::Tensor& target,
    const at::Tensor& weight_value, int64_t reduction, int64_t ignore_index)
{
    at::Tensor weight_tensor = weight_value.defined() ? weight_value : at::ones(self.size(-1), self.options());

    EXEC_NPU_CMD(aclnnNLLLoss, self, target, weight_tensor, reduction, ignore_index, result, total_weight);
    return std::tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}

std::tuple<at::Tensor&, at::Tensor&> nll_loss_forward_out(
    const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, at::Tensor& output, at::Tensor& total_weight)
{
    DO_COMPATIBILITY(aclnnNLLLoss, acl_op::nll_loss_forward_out(self, target, weight, reduction,
                                                                ignore_index, output, total_weight));
    at::Tensor weight_value = c10::value_or_else(weight, [] { return at::Tensor(); });
    if (!weight_value.defined()) {
        weight_value = at::ones(self.size(-1), self.options());
    }

    c10::SmallVector<int64_t, SIZE> output_size = {};
    if (reduction == at::Reduction::None && self.dim() != 1) {
        output_size = {self.size(0)};
    }

    at_npu::native::OpPreparation::check_tensor({self, target, weight_value}, output, output, output_size);
    at_npu::native::OpPreparation::check_tensor({self, target, weight_value}, total_weight, total_weight, {});
    nll_loss_forward_npu_nocheck(output, total_weight, self, target, weight_value, reduction, ignore_index);
    return std::tie(output, total_weight);
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(const at::Tensor& self,
                                                    const at::Tensor& target,
                                                    const c10::optional<at::Tensor>& weight,
                                                    int64_t reduction, int64_t ignore_index)
{
    DO_COMPATIBILITY(aclnnNLLLoss,
                     acl_op::nll_loss_forward(self, target, weight, reduction, ignore_index));
    c10::SmallVector<int64_t, SIZE> output_size = {};
    c10::SmallVector<int64_t, SIZE> totalWeightSize = {};
    const at::Tensor& weight_value = c10::value_or_else(weight, [] { return at::Tensor(); });

    if (reduction == at::Reduction::None && self.dim() != 1) {
        output_size = {self.size(0)};
    }

    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
    at::Tensor total_weight = at_npu::native::OpPreparation::apply_tensor_without_format(self, totalWeightSize);

    nll_loss_forward_npu_nocheck(result, total_weight, self, target, weight_value, reduction, ignore_index);
    return std::tuple<at::Tensor, at::Tensor>(result, total_weight);
}

} // namespace op_api
