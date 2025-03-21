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

#include <ATen/native/LinearAlgebraUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor exec_triangular_solve(
    const at::Tensor& self,
    const at::Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular)
{
    at::Tensor self_broadcasted;
    at::Tensor a_broadcasted;
    std::tie(self_broadcasted, a_broadcasted) = at::native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");
    auto self_working_copy = npu_preparation::apply_tensor(self_broadcasted);
    auto a_working_copy = a_broadcasted.clone();
    EXEC_NPU_CMD(aclnnTriangularSolve, self, A, upper, transpose, unitriangular, self_working_copy, a_working_copy);
    return self_working_copy;
}

} // namespace

at::Tensor& linalg_solve_triangular_out(
    const at::Tensor& self,
    const at::Tensor& B,
    bool upper,
    bool left,
    bool unitriangular,
    at::Tensor& out)
{
    at::Tensor X;
    at::Tensor X_transpose;
    bool transpose = false;
    if (left) {
        X = exec_triangular_solve(B, self, upper, transpose, unitriangular);
        out.resize_as_(X).copy_(X);
    } else {
        X = exec_triangular_solve(B.transpose(-2, -1), self.transpose(-2, -1), !upper, transpose, unitriangular);
        X_transpose = X.transpose(-2, -1);
        out.resize_as_(X_transpose).copy_(X_transpose);
    }
    return out;
}

at::Tensor linalg_solve_triangular(
    const at::Tensor& self,
    const at::Tensor& B,
    bool upper,
    bool left,
    bool unitriangular)
{
    at::Tensor X;
    at::Tensor X_transpose;
    bool transpose = false;
    if (left) {
        X = exec_triangular_solve(B, self, upper, transpose, unitriangular);
        return X;
    } else {
        X = exec_triangular_solve(B.transpose(-2, -1), self.transpose(-2, -1), !upper, transpose, unitriangular);
        X_transpose = X.transpose(-2, -1);
        return X_transpose;
    }
}
}