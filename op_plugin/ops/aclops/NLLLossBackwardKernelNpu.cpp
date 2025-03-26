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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& nll_loss_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight_tensor,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight)
{
    auto scalar_type = target.scalar_type();
    TORCH_CHECK((scalar_type == at::kLong || scalar_type == at::kInt),
                "Expected object of scalar type ", at::kLong, " or ", at::kInt,
                " but got scalar type ", scalar_type, " for argument 'target' in call to nll_loss_backward"
                + OPS_ERROR(ErrCode::TYPE));
    at::Tensor target_cast = (scalar_type == at::kLong) ? at_npu::native::custom_ops::npu_dtype_cast(target, at::kInt) : target;

    string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("NLLLossGrad")
        .Input(self)
        .Input(grad_output)
        .Input(target_cast)
        .Input(weight_tensor)
        .Input(total_weight)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Attr("ignore_index", ignore_index)
        .Run();

    if (self.dim() == 1) {
        grad_input.squeeze_(0);
    }

    return grad_input;
}
} // namespace

at::Tensor& nll_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input)
{
    at::Tensor self_cp = self.dim() == 1 ? self.unsqueeze(0) : self;
    const at::Tensor& weight_value_or = c10::value_or_else(weight, [] {return at::Tensor();});
    at::Tensor weight_tensor = at::ones(self_cp.size(1), self_cp.options());
    if (weight_value_or.defined()) {
        weight_tensor = npu_utils::format_contiguous(weight_value_or);
    }

    if (ignore_index >= 0 && ignore_index < self_cp.size(-1)) {
        at::Tensor zero = at::zeros(1, self_cp.options());
        calcu_op_util::AclrtMemcpyAsync(
            {weight_tensor, ignore_index},
            weight_tensor.itemsize(),
            {zero, 0},
            weight_tensor.itemsize(),
            ACL_MEMCPY_DEVICE_TO_DEVICE);
    }

    npu_preparation::CheckOut(
        {self_cp},
        grad_input,
        self_cp);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
        nll_loss_backward_out_nocheck(contiguous_grad_input, grad_output, self_cp, target, weight_tensor, reduction,
                                      ignore_index, total_weight);
        npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
    } else {
        nll_loss_backward_out_nocheck(grad_input, grad_output, self_cp, target, weight_tensor, reduction, ignore_index,
                                      total_weight);
    }

    return grad_input;
}

at::Tensor nll_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    acl_op::nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index,
                                  total_weight, grad_input);
    return grad_input;
}
} // namespace acl_op
