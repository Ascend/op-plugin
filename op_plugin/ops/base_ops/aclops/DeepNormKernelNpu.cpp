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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_deep_norm(const at::Tensor& x,
                                                             const at::Tensor& gx,
                                                             const at::Tensor& beta,
                                                             const at::Tensor& gamma,
                                                             double alpha,
                                                             double epsilon)
{
    at::SmallVector<int64_t, SIZE> shape;
    auto param_dim = x.dim() - gamma.dim();
    for (uint64_t index = 0; index < x.dim(); index++) {
        if (index < param_dim) {
            shape.emplace_back(x.size(index));
        } else {
            shape.emplace_back(1);
        }
    }
    at::Tensor y = npu_preparation::apply_tensor(x);
    at::Tensor mean = npu_preparation::apply_tensor(shape, x.options().dtype(at::kFloat), x);
    at::Tensor rstd = npu_preparation::apply_tensor(shape, x.options().dtype(at::kFloat), x);
    at_npu::native::OpCommand cmd;
    cmd.Name("DeepNorm")
        .Input(x, "x")
        .Input(gx, "gx")
        .Input(beta, "beta")
        .Input(gamma, "gamma")
        .Output(mean, "mean")
        .Output(rstd, "rstd")
        .Output(y, "y")
        .Attr("alpha", static_cast<float>(alpha))
        .Attr("epsilon", static_cast<float>(epsilon))
        .Run();

    return std::make_tuple(mean, rstd, y);
}
} // namespace acl_op