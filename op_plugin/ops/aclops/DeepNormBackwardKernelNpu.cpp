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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deep_norm_backward(const at::Tensor& dy,
                                                                                  const at::Tensor& x,
                                                                                  const at::Tensor& gx,
                                                                                  const at::Tensor& gamma,
                                                                                  const at::Tensor& mean,
                                                                                  const at::Tensor& rstd,
                                                                                  double alpha)
{
    at::Tensor dx = npu_preparation::apply_tensor(x);
    at::Tensor dgx = npu_preparation::apply_tensor(gx);
    at::Tensor dbeta = npu_preparation::apply_tensor(gamma.sizes(), gamma.options().dtype(at::kFloat), gamma);
    at::Tensor dgamma = npu_preparation::apply_tensor(gamma.sizes(), gamma.options().dtype(at::kFloat), gamma);
    at_npu::native::OpCommand cmd;
    cmd.Name("DeepNormGrad")
        .Input(dy, "dy")
        .Input(x, "x")
        .Input(gx, "gx")
        .Input(gamma, "gamma")
        .Input(mean, "mean")
        .Input(rstd, "rstd")
        .Output(dx, "dx")
        .Output(dgx, "dgx")
        .Output(dbeta, "dbeta")
        .Output(dgamma, "dgamma")
        .Attr("alpha", static_cast<float>(alpha))
        .Run();

    return std::make_tuple(dx, dgx, dbeta, dgamma);
}
} // namespace acl_op

