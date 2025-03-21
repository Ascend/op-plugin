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

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor kl_div_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target)
{
    auto output_size = op_infer::input_same_output_size(self);
    at::Tensor grad_input = npu_preparation::apply_tensor(output_size, self.options(), self);
    std::string reduction_str;
    if (reduction == at::Reduction::Mean) {
        reduction_str = "batchmean";
    } else if (reduction == at::Reduction::Sum) {
        reduction_str = "sum";
    } else if (reduction == at::Reduction::None) {
        reduction_str = "none";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("KlDivLossGrad")
        .Input(grad_output)
        .Input(self)
        .Input(target)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Attr("log_target", log_target)
        .Run();
    if (reduction == at::Reduction::Mean) {
        auto input_shape = self.sizes();
        int batch_square_size = c10::multiply_integers(input_shape) / input_shape[0];
        grad_input.div_(batch_square_size);
    }
    return grad_input;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor kl_div_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target)
{
    auto output_size = op_infer::input_same_output_size(self);
    at::Tensor grad_input = npu_preparation::apply_tensor(output_size, self.options(), self);
    std::string reduction_str = "none";
    if (reduction == at::Reduction::Mean) {
        reduction_str = "batchmean";
    } else if (reduction == at::Reduction::Sum) {
        reduction_str = "sum";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("KlDivLossGrad")
        .Input(grad_output)
        .Input(self)
        .Input(target)
        .Output(grad_input)
        .Attr("reduction", reduction_str)
        .Attr("log_target", log_target)
        .Run();
    if (reduction == at::Reduction::Mean) {
        auto input_shape = self.sizes();
        int batch_square_size = input_shape.size() > 1 ? c10::multiply_integers(input_shape.slice(1)) : 1;
        grad_input.div_(batch_square_size);
    }
    return grad_input;
}
#endif
} // namespace acl_op
