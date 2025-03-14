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
at::Tensor& binary_cross_entropy_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction)
{
    at::Tensor weight_tensor = weight.defined() ? weight : at::ones(self.sizes(), self.options());
    std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("BinaryCrossEntropyGrad")
        .Input(self)
        .Input(target)
        .Input(grad_output)
        .Input(weight_tensor)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor& binary_cross_entropy_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& grad_input)
{
    const at::Tensor& weight_value = c10::value_or_else(weight, [] {return at::Tensor();});
    npu_preparation::CheckOut(
        {grad_output, self, target, weight_value},
        grad_input,
        self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        binary_cross_entropy_backward_out_npu_nocheck(contiguous_result, grad_output, self, target, weight_value, reduction);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        binary_cross_entropy_backward_out_npu_nocheck(grad_input, grad_output, self, target, weight_value, reduction);
    }
    return grad_input;
}

at::Tensor binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction)
{
    const at::Tensor& weight_value = c10::value_or_else(weight, [] {return at::Tensor();});
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    binary_cross_entropy_backward_out_npu_nocheck(grad_input, grad_output, self, target, weight_value, reduction);
    return grad_input;
}
} // namespace acl_op
