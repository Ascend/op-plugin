// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> _linalg_svd_out(
    const at::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver,
    at::Tensor& U,
    at::Tensor& S,
    at::Tensor& Vh)
{
    DO_COMPATIBILITY(aclnnSvd, acl_op::_linalg_svd_out(A, full_matrices, compute_uv, driver, U, S, Vh));
    if (A.numel() == 0) {
        if (compute_uv && full_matrices) {
            if (U.numel() != 0) {
                U.zero_();
                U.diagonal(0, -2, -1).fill_(1.);
            }
            if (Vh.numel() != 0) {
                Vh.zero_();
                Vh.diagonal(0, -2, -1).fill_(1.);
            }
        }
        return std::forward_as_tuple(U, S, Vh);
    }
    auto V = Vh.dim() <= 1 ? Vh : Vh.transpose(-1, -2);
    EXEC_NPU_CMD(aclnnSvd, A, full_matrices, compute_uv, S, U, V);
    return std::forward_as_tuple(U, S, Vh);
}
}