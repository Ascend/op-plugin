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
at::Tensor threshold_backward_out_npu(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar threshold)
{
    at_npu::native::OpCommand cmd;
    // The performance of the ReluGrad operator is better than that of ThresholdGradV2D.
    // However, ReluGrad does not support the scenario where threshold is not 0.
    if (op_plugin::utils::get_scalar_float_value(threshold) != static_cast<float>(0)) {
        cmd.Name("ThresholdGradV2D")
            .Input(grad_output)
            .Input(self)
            .Output(result)
            .Attr("threshold", threshold)
            .Run();
    } else {
        cmd.Name("ReluGrad")
            .Input(grad_output)
            .Input(self)
            .Output(result)
            .Run();
    }

    return result;
}
} // namespace

at::Tensor threshold_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    threshold_backward_out_npu(result, grad_output, self, threshold);
    return result;
}
} // namespace acl_op
