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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& l1_loss_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const int64_t reduction)
{
    at::Tensor grad_output_broadcast =
        grad_output.sizes() != self.sizes() ? acl_op::npu_broadcast(grad_output, self.sizes()) : grad_output;
    at::Tensor target_broadcast =
        target.sizes() != self.sizes() ? acl_op::npu_broadcast(target, self.sizes()) : target;

    std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("L1LossGrad")
        .Input(grad_output_broadcast)
        .Input(self)
        .Input(target_broadcast)
        .Attr("reduction", reduction_str)
        .Output(grad_input)
        .Run();
    return grad_input;
}
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& l1_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {grad_output, self, target},
        grad_input,
        self);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
        l1_loss_backward_out_nocheck(grad_input, grad_output, self, target, reduction);
        npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
    } else {
        l1_loss_backward_out_nocheck(grad_input, grad_output, self, target, reduction);
    }
    return grad_input;
}

at::Tensor l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    l1_loss_backward_out_nocheck(result, grad_output, self, target, reduction);
    return result;
}
#endif
} // namespace acl_op
