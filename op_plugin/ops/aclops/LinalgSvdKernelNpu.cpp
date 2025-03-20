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
#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> _linalg_svd_out(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    at::Tensor& U,
    at::Tensor& S,
    at::Tensor& Vh)
{
    return linalg_svd_out_common(A, full_matrices, compute_uv, U, S, Vh);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _linalg_svd(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv)
{
    return _svd_helper(A, !full_matrices, compute_uv);
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> _linalg_svd_out(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver,
    at::Tensor& U,
    at::Tensor& S,
    at::Tensor& Vh)
{
    return linalg_svd_out_common(A, full_matrices, compute_uv, U, S, Vh);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _linalg_svd(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver)
{
    return _svd_helper(A, !full_matrices, compute_uv);
}
#endif

} // namespace acl_op
