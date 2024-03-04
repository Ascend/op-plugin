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

#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

#include <ATen/native/LinearAlgebraUtils.h>

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> triangular_solve_out_common_nocheck(const at::Tensor &self, const at::Tensor &A,
                                                                       bool upper, bool transpose, bool unitriangular)
{
    at::Tensor self_broadcasted;
    at::Tensor a_broadcasted;
    std::tie(self_broadcasted, a_broadcasted) = at::native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");
    TORCH_CHECK(self_broadcasted.dtype() == at::kFloat && a_broadcasted.dtype() == at::kFloat,
                "_triangular_solve_helper_npu only supported Float, but get ", self_broadcasted.dtype(), ' ',
                a_broadcasted.dtype(), OPS_ERROR(ErrCode::TYPE));
    auto self_working_copy = npu_preparation::apply_tensor(self_broadcasted);
    auto a_working_copy = a_broadcasted.clone();
    at::Tensor a_tensor = a_broadcasted;
    TORCH_CHECK(a_tensor.dim() >= 2, "The dim of input tensor must larger than two.", OPS_ERROR(ErrCode::VALUE));
    if (unitriangular) {
        auto diagonal_tensor = at::eye(a_tensor.size(-2), a_tensor.size(-1), a_tensor.options());
        a_tensor = a_tensor * (1 - diagonal_tensor) + diagonal_tensor;
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("MatrixTriangularSolve")
        .Input(a_tensor)
        .Input(self_broadcasted)
        .Output(self_working_copy)
        .Attr("lower", !upper)
        .Attr("adjoint", transpose)
        .Run();

    return std::tie(self_working_copy, a_working_copy);
}
} // namespace acl_op
