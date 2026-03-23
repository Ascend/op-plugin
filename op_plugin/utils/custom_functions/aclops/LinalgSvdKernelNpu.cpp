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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

int64_t batch_count(const at::Tensor &batched_matrices)
{
    int64_t result = 1;
    auto number = 2;
    for (int64_t i = 0; i < batched_matrices.ndimension() - number; i++) {
        result *= batched_matrices.size(i);
    }
    return result;
}

void single_check_errors(int64_t info, const char *name, bool allow_singular = false, int64_t batch_idx = -1)
{
    std::string batch_info = "";
    if (batch_idx >= 0) {
        batch_info = ": For batch " + std::to_string(batch_idx);
    }
    if (info < 0) {
        TORCH_CHECK(false, name, batch_info, ": Argument ", -info, " has illegal value", OPS_ERROR(ErrCode::VALUE));
    } else if (info > 0) {
        TORCH_CHECK(!strstr(name, "svd"), name, ": the updating process of SBDSDC did not converge (error: ", info,
                    ")", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(!strstr(name, "symeig"), name, batch_info, ": the algorithm failed to converge; ", info,
                    " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.",
                    OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(allow_singular, name, batch_info, ": U(", info, ",", info, ") is zero, singular U.",
                    OPS_ERROR(ErrCode::PARAM));
    }
}

void batch_check_errors(std::vector<int64_t> &infos, const char *name, bool allow_singular = false)
{
    for (size_t i = 0; i < infos.size(); i++) {
        auto info = infos[i];
        single_check_errors(info, name, allow_singular, i);
    }
}

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[B * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[B * M' * N'].
 */
inline at::Tensor cloneBatchedColumnMajor(const at::Tensor &src)
{
    // If src is already in batched column major format, then
    // this will be efficient (no reordering of the data will occur)
    // because the first transpose will make the tensor contiguous,
    // and cloning a contiguous tensor is fast.
    auto result = src.mT().clone(at::MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
    return result;
}

/*
 * contig chooses between C-contig (true) and F-contig (false)
 */
inline c10::MaybeOwned<at::Tensor> borrow_else_clone(const bool cond, const at::Tensor &borrow, const at::Tensor &clone,
                                                     const bool contig)
{
    return cond ? c10::MaybeOwned<at::Tensor>::borrowed(borrow) :
                  c10::MaybeOwned<at::Tensor>::owned(contig ? clone.clone(at::MemoryFormat::Contiguous) :
                                                              cloneBatchedColumnMajor(clone));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _svd_helper(const at::Tensor &self, bool some, bool compute_uv)
{
    TORCH_CHECK(self.dtype() == at::kFloat, "svd_npu only supported Float, but get", self.dtype(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(self.dim() >= 2, "The dim of input tensor must larger than two.", OPS_ERROR(ErrCode::VALUE));
    std::vector<int64_t> infos(batch_count(self), 0);
    int64_t m = self.size(-2);
    int64_t n = self.size(-1);
    int64_t k = std::min(m, n);

    at::Tensor U_working_copy;
    at::Tensor S_working_copy;
    at::Tensor VT_working_copy;
    auto sizes = self.sizes().vec();

    auto number_a = 2;
    auto number_b = 1;
    sizes[self.dim() - number_b] = (compute_uv && some) ? std::min(m, n) : m;
    U_working_copy = npu_preparation::apply_tensor(self, sizes);

    sizes[self.dim() - number_a] = n;
    sizes[self.dim() - number_b] = (compute_uv && some) ? k : n;
    VT_working_copy = npu_preparation::apply_tensor(self, sizes);

    sizes.pop_back();
    sizes[self.dim() - number_a] = std::min(m, n);
    S_working_copy = npu_preparation::apply_tensor(self, sizes);

    if (self.numel() > 0) {
        at_npu::native::OpCommand cmd;
        cmd.Name("Svd")
            .Input(self)
            .Output(S_working_copy)
            .Output(U_working_copy)
            .Output(VT_working_copy)
            .Attr("compute_uv", compute_uv)
            .Attr("full_matrices", !some)
            .Run();

        if (self.dim() > number_a) {
            batch_check_errors(infos, "svd_npu");
        } else {
            single_check_errors(infos[0], "svd_npu");
        }

        if (!compute_uv) {
            VT_working_copy.zero_();
            U_working_copy.zero_();
        }
    } else {
        U_working_copy.zero_();
        VT_working_copy.zero_();
        if (compute_uv && !some) {
            U_working_copy.diagonal(0, -2, -1).fill_(1.);
            VT_working_copy.diagonal(0, -2, -1).fill_(1.);
        }
    }
    return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy.mH());
}

static void linalg_check_errors(const at::Tensor &infos, const c10::string_view api_name, bool is_matrix)
{
    TORCH_CHECK(infos.scalar_type() == at::kInt, OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(infos.is_contiguous(), OPS_ERROR(ErrCode::VALUE));
    if (infos.is_meta()) {
        return;
    }

    // If it's all zeros, we return early.
    // We optimise for the most likely case.
    if (C10_LIKELY(!infos.to(at::kBool).any().item<bool>())) {
        return;
    }

    int32_t info;
    std::string batch_str;
    if (is_matrix) {
        info = infos.item<int>();
        // batch_str needn't be set for matrices
    } else {
        // Find the first non-zero info
        auto infos_cpu = infos.to(at::kCPU);
        auto ptr = infos_cpu.data_ptr<int32_t>();
        TORCH_CHECK(ptr != nullptr, "infos is null", OPS_ERROR(ErrCode::PTR))
        auto n = infos.numel();
        auto info_ptr = std::find_if(ptr, ptr + n, [](int32_t x) { return x != 0; });
        info = *info_ptr;
        batch_str = ": (Batch element " + std::to_string(std::distance(ptr, info_ptr)) + ")";
    }

    if (info < 0) {
        // Reference LAPACK 3.10+ changed `info` behavior for inputs with non-finite values
        // Previously, it would return `info` > 0, but now it returns `info` = -4
        // OpenBLAS 0.3.15+ uses the Reference LAPACK 3.10+.
        // MKL 2022.0+ uses the Reference LAPACK 3.10+.
        // Older version of MKL and OpenBLAS follow the old behavior (return `info` > 0).
        // Here we check for the case where `info` is -4 and raise an error
        if (api_name.find("svd") != api_name.npos) {
            TORCH_CHECK(info != -4, api_name, batch_str,
                        ": The algorithm failed to converge because the input matrix contained non-finite values.",
                        OPS_ERROR(ErrCode::VALUE));
        }
        TORCH_CHECK(
            false, api_name, batch_str, ": Argument ", -info,
            " has illegal value. Most certainly there is a bug in the implementation calling the backend library.",
            OPS_ERROR(ErrCode::VALUE));
    } else if (info > 0) {
        if (api_name.find("svd") != api_name.npos) {
            TORCH_CHECK(false, api_name, batch_str,
                        ": The algorithm failed to converge because the input matrix is ill-conditioned or has too "
                        "many repeated singular values (error code: ",
                        info, ").", OPS_ERROR(ErrCode::PARAM));
        } else {
            TORCH_CHECK(false, api_name, ": Unknown error code: ", info, ".", OPS_ERROR(ErrCode::INTERNAL));
        }
    }
    // We should never reach this point as info was non-zero
    TORCH_CHECK(false, OPS_ERROR(ErrCode::INTERNAL));
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> linalg_svd_out_common(const at::Tensor &A,
                                                                           const bool full_matrices,
                                                                           const bool compute_uv, at::Tensor &U,
                                                                           at::Tensor &S, at::Tensor &Vh)
{
    // Half optimisation half precondition for some parts of the LAPACK / cuSOLVER
    // In particular, the call to lapackSvd to compute lwork fails otherwise
    if (A.numel() == 0) {
        // Needed in the case that we have e.g. A.shape == (3, 0) and full_matrices=True
        // We fill U or Vh with the identity matrix as it's a valid SVD for the empty matrix
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
        return std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>(U, S, Vh);
    }

    const bool use_cusolver = false;

    // A always needs to be copied as its contents will be destroyed during the computaton of the SVD
    // Now, MAGMA needs the copy to be on CPU, while cuSOLVER needs it to be on CUDA, so we'll defer
    // the copy as a column major matrix to the backends.
    const auto info = at::zeros(at::IntArrayRef(A.sizes().begin(), A.sizes().end() - 2), A.options().dtype(at::kInt));

    // Prepare S
    const auto S_ = S.expect_contiguous();

    // Prepare U / Vh
    // U_ and Vh_ are just going to be accessed whenever compute_uv == true
    const auto U_ready = !compute_uv || U.mT().is_contiguous();
    const auto U_ = borrow_else_clone(U_ready, U, U, false);
    const auto Vh_ready =
        !compute_uv || (!use_cusolver && Vh.mT().is_contiguous()) || (use_cusolver && Vh.is_contiguous());
    const auto Vh_ = borrow_else_clone(Vh_ready, Vh, Vh, use_cusolver);

    at::Tensor U_tmp;
    at::Tensor S_tmp;
    at::Tensor V_tmp;
    std::tie(U_tmp, S_tmp, V_tmp) = _svd_helper(A, !full_matrices, compute_uv);
    if (!U_ready) {
        U.copy_(U_tmp);
    }
    if (!S.is_same(S_tmp)) {
        S.copy_(S_tmp);
    }
    if (!Vh_ready) {
        Vh.copy_(V_tmp);
    }
    linalg_check_errors(info, "linalg.svd", A.dim() == 2);
    return std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>(U, S, Vh);
}
} // namespace acl_op
