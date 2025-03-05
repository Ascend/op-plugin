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

namespace {
std::tuple<at::Tensor&, at::Tensor&> softmax_cross_entropy_with_logits_out_nocheck(
    at::Tensor& result,
    at::Tensor& backprop,
    const at::Tensor& self,
    const at::Tensor& labels)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SoftmaxCrossEntropyWithLogits")
        .Input(self)
        .Input(labels)
        .Output(result)
        .Output(backprop)
        .Run();

    return std::tuple<at::Tensor&, at::Tensor&>(result, backprop);
}

std::tuple<at::Tensor, at::Tensor> softmax_cross_entropy_with_logits_impl_out_nocheck(
    const at::Tensor& self,
    const at::Tensor& labels)
{
    auto output_sizes = op_infer::softmax_cross_entropy_with_logits_impl_npu_output_size(self);
    at::Tensor result = npu_preparation::apply_tensor(self, std::get<0>(output_sizes));
    at::Tensor backprop = npu_preparation::apply_tensor(self, std::get<1>(output_sizes));

    softmax_cross_entropy_with_logits_out_nocheck(result, backprop, self, labels);

    return std::make_tuple(result, backprop);
}
} // namespace

at::Tensor npu_softmax_cross_entropy_with_logits_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& labels)
{
    at::Tensor result1 = std::get<1>(softmax_cross_entropy_with_logits_impl_out_nocheck(self, labels));
    return result1 * grad.unsqueeze(-1);
}


at::Tensor npu_softmax_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& labels)
{
    TORCH_CHECK(torch_npu::utils::is_npu(self), OPS_ERROR(ErrCode::PARAM));
    return std::get<0>(softmax_cross_entropy_with_logits_impl_out_nocheck(self, labels));
}
} // namespace acl_op
