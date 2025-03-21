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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return at::Tensor();});
    at::Tensor weight_tensor;
    if (weight.defined()) {
        weight_tensor = npu_utils::format_contiguous(weight);
        weight_tensor = (weight_tensor.scalar_type() != self.scalar_type()) ?
            at_npu::native::custom_ops::npu_dtype_cast(weight_tensor, self.scalar_type()) : weight_tensor;
    } else {
        weight_tensor = at::ones(self.sizes(), self.options());
    }

    at::Tensor pos_weight_tensor;
    if (pos_weight.defined()) {
        pos_weight_tensor = npu_utils::format_contiguous(pos_weight);
        pos_weight_tensor = (pos_weight_tensor.scalar_type() != self.scalar_type()) ?
            at_npu::native::custom_ops::npu_dtype_cast(pos_weight_tensor, self.scalar_type()) : pos_weight_tensor;
    } else {
        pos_weight_tensor = at::ones(self.sizes(), self.options());
    }

    at::Tensor dout_tensor = acl_op::npu_broadcast(grad_output, self.sizes());
    std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("SigmoidCrossEntropyWithLogitsGradV2")
        .Input(self)
        .Input(target)
        .Input(dout_tensor)
        .Input(weight_tensor)
        .Input(pos_weight_tensor)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Run();

    return grad_input;
}
#endif
} // namespace acl_op
