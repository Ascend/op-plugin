// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>npu_add_layer_norm_backward(const at::Tensor &dy,
                                                                                      const at::Tensor &x1,
                                                                                      const at::Tensor &x2,
                                                                                      const at::Tensor &rstd,
                                                                                      const at::Tensor &mean,
                                                                                      const at::Tensor &gamma)
{
    at::SmallVector<int64_t, SIZE> shape;
    shape.emplace_back(gamma.size(0));

    at::Tensor dx = npu_preparation::apply_tensor(dy);
    at::Tensor dgamma = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor dbeta = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);

    at_npu::native::OpCommand cmd;

    cmd.Name("AddLayerNormGrad")
        .Input(dy, "dy")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(rstd, "rstd")
        .Input(mean, "mean")
        .Input(gamma, "gamma")
        .Output(dx, "dx")
        .Output(dgamma, "dgamma")
        .Output(dbeta, "dbeta")
        .Run();

    return std::make_tuple(dx, dx, dgamma, dbeta);
}
} // namespace acl_op