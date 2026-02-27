// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<c10::SmallVector<int64_t, N>, c10::SmallVector<int64_t, N>> qr_npu_output_size(
    const at::Tensor& self,
    bool some)
{
    int m = self.size(-2);
    int n = self.size(-1);
    auto k = std::min<int>(m, n);
    auto shape = op_infer::array_to_small_vector(self.sizes());
    c10::SmallVector<int64_t, N> q_size(shape.begin(), shape.end() - 2);
    c10::SmallVector<int64_t, N> r_size(shape.begin(), shape.end() - 2);
    if(some){
        q_size.insert(q_size.end(), {m, k});
        r_size.insert(r_size.end(), {k, n});
    } else {
        q_size.insert(q_size.end(), {m, m});
        r_size.insert(r_size.end(), {m, n});
    }
    return std::tie(q_size, r_size);
}

inline bool mode_valid(c10::string_view mode)
{
    return (mode == "reduced" || mode == "complete" || mode == "r");
}

void check_linalg_qr_input(const at::Tensor& self, c10::string_view mode)
{
    constexpr int MATRIX_DIM = 2;
    TORCH_CHECK(
        self.dim() >= MATRIX_DIM,
        "linalg_qr: The input tensor must have at least 2 dimensions, but got ",
        self.dim(),
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(
        mode_valid(mode),
        "linalg_qr: received unrecognized mode '",
        mode,
        "', expected one of 'reduced'(default), 'r', or 'complete'",
        OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&> qr_out_npu_nocheck(
    at::Tensor& Q,
    at::Tensor& R,
    const at::Tensor& self,
    bool some)
{
    bool full_matrices = !some;
    at_npu::native::OpCommand cmd;
    cmd.Name("Qr")
        .Input(self)
        .Output(Q)
        .Output(R)
        .Attr("full_matrices", full_matrices)
        .Run();
    return std::tie(Q, R);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> linalg_qr_out(
    const at::Tensor& self,
    c10::string_view mode,
    at::Tensor& Q,
    at::Tensor& R)
{
    check_linalg_qr_input(self, mode);
    bool some = (mode == "complete") ? false : true;
    auto sizes = qr_npu_output_size(self, some);
    npu_preparation::CheckOut(
        {self},
        Q,
        self,
        std::get<0>(sizes));
    npu_preparation::CheckOut(
        {self},
        R,
        self,
        std::get<1>(sizes));
    bool q_match = npu_utils::check_match(&Q);
    bool r_match = npu_utils::check_match(&R);
    if (!(q_match && r_match)) {
        at::Tensor contiguous_q = q_match ? Q : npu_utils::format_contiguous(Q);
        at::Tensor contiguous_r = r_match ? R : npu_utils::format_contiguous(R);
        qr_out_npu_nocheck(contiguous_q, contiguous_r, self, some);
        if (!q_match) {
            npu_utils::format_fresh_view(Q, contiguous_q);
        }
        if (!r_match) {
            npu_utils::format_fresh_view(R, contiguous_r);
        }
    } else {
        qr_out_npu_nocheck(Q, R, self, some);
    }
    if (mode == "r") {
        c10::SmallVector<int64_t, op_infer::N> Esize = {0};
        npu_preparation::CheckOut({self}, Q, self, Esize);
    }
    return std::tie(Q, R);
}

std::tuple<at::Tensor, at::Tensor> linalg_qr(
    const at::Tensor& self,
    c10::string_view mode)
{
    check_linalg_qr_input(self, mode);
    bool some = (mode == "complete") ? false : true;
    auto sizes = qr_npu_output_size(self, some);
    at::Tensor Q = npu_preparation::apply_tensor(self, std::get<0>(sizes));
    at::Tensor R = npu_preparation::apply_tensor(self, std::get<1>(sizes));

    qr_out_npu_nocheck(Q, R, self, some);
    if (mode == "r") {
        c10::SmallVector<int64_t, op_infer::N> Esize = {0};
        Q = npu_preparation::apply_tensor_without_format(Esize, self.options());
    }
    return std::tie(Q, R);
}
} // namespace acl_op
