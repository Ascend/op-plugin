// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

void _foreach_exp_(const at::TensorList self)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_exp_slow_(self);
    }

    if (self.empty()) {
        return;
    }
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float");
    }
    EXEC_NPU_CMD(aclnnForeachExp, self, self);
}


std::vector<at::Tensor> _foreach_exp(const at::TensorList self)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_exp_slow(self);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float");
    }

    // construct output tensorlist
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    EXEC_NPU_CMD(aclnnForeachExp, self, result_);
    return result;
}
} // namespace at_npu

