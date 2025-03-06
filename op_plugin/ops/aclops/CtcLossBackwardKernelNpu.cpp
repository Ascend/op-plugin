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

at::Tensor _ctc_loss_backward(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity)
{
    at::Tensor grad_out_cast = grad.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(grad, at::kFloat) : grad;
    at::Tensor log_probs_cast = log_probs.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(log_probs, at::kFloat) : log_probs;
    at::Tensor neg_log_likelihood_cast = neg_log_likelihood.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(neg_log_likelihood, at::kFloat) : neg_log_likelihood;
    at::Tensor log_alpha_cast = log_alpha.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(log_alpha, at::kFloat) : log_alpha;

    auto input_lengths_tensor = at::tensor(input_lengths, targets.options());
    auto target_lengths_tensor = at::tensor(target_lengths, targets.options());
    at::Tensor grad_out = npu_preparation::apply_tensor(log_probs_cast);

    at_npu::native::OpCommand cmd;
    cmd.Name("CTCLossV2Grad")
        .Input(grad_out_cast)
        .Input(log_probs_cast)
        .Input(targets)
        .Input(input_lengths_tensor)
        .Input(target_lengths_tensor)
        .Input(neg_log_likelihood_cast)
        .Input(log_alpha_cast)
        .Output(grad_out)
        .Attr("blank", blank)
        .Attr("zero_infinity", zero_infinity)
        .Run();

    if (grad.scalar_type() == at::kHalf) {
        grad_out = at_npu::native::custom_ops::npu_dtype_cast(grad_out, at::kHalf);
    }
    return grad_out;
}
} // namespace acl_op
