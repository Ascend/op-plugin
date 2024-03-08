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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor bincount(const at::Tensor &self, const c10::optional<at::Tensor> &weight_opt, int64_t minlength)
{
    const at::Tensor &weights = c10::value_or_else(weight_opt, [] { return at::Tensor(); });
    TORCH_CHECK(self.numel() > 0, "The input tensor is an empty tensor." + OPS_ERROR(ErrCode::PARAM));
    if (self.sizes()[0] == 0) {
        auto result = npu_preparation::ApplyTensorWithSizes({0}, self.options().dtype(at::kLong));
        return result;
    }

    auto sizes = static_cast<int64_t>(op_plugin::utils::get_scalar_float_value(acl_op::max(self).item()));
    sizes = (sizes < minlength) ? minlength : (sizes + 1);

    if (self.dtype() == at::kLong) {
        TORCH_NPU_WARN_ONCE("CANN: Bincount cann't support dtype int64, input will be cast to int32.");
    }
    auto input = (self.dtype() == at::kInt) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt);

    // weight convert dtype as same as output defined by torch
    auto weight = weights;
    if (!weights.defined()) {
        at::TensorOptions options = input.options();
        weight = acl_op::ones(input.sizes(), at::kLong, options.layout(), options.device(), options.pinned_memory());
    } else if (!(weights.dtype() == at::kFloat)) {
        weight = at_npu::native::custom_ops::npu_dtype_cast(weights, at::kDouble);
    }
    auto result = npu_preparation::apply_tensor(weight, {sizes});

    at_npu::native::OpCommand cmd;
    cmd.Name("Bincount").Input(input).Input(at::Scalar(sizes), at::kInt).Input(weight).Output(result).Run();

    return result;
}
} // namespace acl_op
