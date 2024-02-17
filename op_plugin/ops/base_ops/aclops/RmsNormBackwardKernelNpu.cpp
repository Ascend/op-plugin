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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_backward(const at::Tensor &dy, const at::Tensor &self,
                                                         const at::Tensor &gamma, const at::Tensor &rstd)
{
    at::Tensor dx = npu_preparation::apply_tensor(self);
    at::Tensor dgamma = npu_preparation::apply_tensor(gamma.sizes(), gamma.options().dtype(at::kFloat), gamma);
    at_npu::native::OpCommand cmd;
    cmd.Name("RmsNormGrad")
        .Input(dy, "dy")
        .Input(self, "x")
        .Input(rstd, "rstd")
        .Input(gamma, "gamma")
        .Output(dx, "dx")
        .Output(dgamma, "dgamma")
        .Run();

    return std::make_tuple(dx, dgamma);
}
} // namespace acl_op
