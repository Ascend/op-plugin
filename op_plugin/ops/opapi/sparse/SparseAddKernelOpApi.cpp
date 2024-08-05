// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include <ATen/native/SparseTensorUtils.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "op_plugin/SparseOpsInterface.h"
#include "SparseTensorUtils.h"

namespace sparse {
at::Tensor& add_out_dense_sparse_npu(
    at::Tensor& r_,
    const at::Tensor& dense,
    const at::sparse::SparseTensor& sparse,
    const at::Scalar& value)
{
    TORCH_CHECK(torch_npu::utils::is_npu(dense), "sparse add: expected 'self' to be a NPU tensor",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(torch_npu::utils::is_npu(sparse), "sparse add: expected 'other' to be a NPU tensor",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(torch_npu::utils::is_npu(r_), "sparse add: expected 'out' to be a NPU tensor",
        OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(dense.sizes().equals(sparse.sizes()),
        "add: expected 'self' and 'other' to have same size, but self has size ",
        dense.sizes(), " while other has size ", sparse.sizes(),
        " (FYI: dense-sparse addition does not currently support broadcasting)", OPS_ERROR(ErrCode::VALUE));

    const int64_t nnz = sparse._nnz();
    if (nnz == 0) {
        r_.resize_as_(dense);
        r_.copy_(dense);
        return r_;
    }

    auto common_dtype = at::result_type(dense, sparse);
    TORCH_CHECK(c10::canCast(common_dtype, r_.scalar_type()), "Can't convert result type ", common_dtype,
        " to output ", r_.scalar_type(), OPS_ERROR(ErrCode::TYPE));

    at::Tensor r = r_;
    if (r_.scalar_type() != common_dtype) {
        r = at::empty_like(dense, r_.options().dtype(common_dtype));
    }

    at::Tensor dense_buffer = dense.to(common_dtype);
    at::Tensor values = sparse._values().to(common_dtype);

    if (at::sparse::is_same_tensor(r, dense_buffer)) {
        TORCH_CHECK(r_.is_contiguous(),
            "add: NPU dense-sparse addition with a non-contiguous output tensor does not work",
            OPS_ERROR(ErrCode::VALUE));
    } else {
        r.resize_as_(dense);
        r.copy_(dense_buffer);
    }

    at::Tensor indices = sparse._indices();
    int64_t n_dim = dense.dim();
    int64_t n_dimI = sparse.sparse_dim();

    if (values.numel() == 0) {
        return r_;
    }

    at::Tensor indices_1D = at::sparse::flatten_indices(indices, sparse.sizes(), false);

    int64_t view_rows = 1;
    int64_t view_columns = 1;
    for (int i = 0; i < n_dimI; i++) {
        view_rows *= r.size(i);
    }
    for (int i = n_dimI; i < n_dim; i++) {
        view_columns *= r.size(i);
    }

    at::Tensor r_view = r.view({view_rows, view_columns});
    values = values.reshape({nnz, view_columns});
    r_view.index_add_(0, indices_1D, values, value);

    r_.copy_(r);
    return r_;
}


at::sparse::SparseTensor& add_out_sparse(
    const at::sparse::SparseTensor& t,
    const at::sparse::SparseTensor& src,
    const at::Scalar& value,
    at::sparse::SparseTensor& r_)
{
    if (!t.is_sparse()) {
        return add_out_dense_sparse_npu(r_, t, src, value);
    }

    TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.",
        OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(torch_npu::utils::is_npu(t), "add: expected 'self' to be NPU, but got ", t.device(),
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(torch_npu::utils::is_npu(src), "add: expected 'other' to be NPU, but got ", src.device(),
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(torch_npu::utils::is_npu(r_), "add: expected 'out' to be NPU, but got ", r_.device(),
        OPS_ERROR(ErrCode::VALUE));

    auto common_dtype = at::result_type(t, src);
    TORCH_CHECK(c10::canCast(common_dtype, r_.scalar_type()), "Can't convert result type ",
        common_dtype, " to output ", r_.scalar_type(), OPS_ERROR(ErrCode::TYPE));

    TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but ",
        t.sizes(), " != ", src.sizes(), OPS_ERROR(ErrCode::VALUE));

    if (src._nnz() == 0) {
        return at::copy_sparse_to_sparse_(r_, t);
    }
    if (t._nnz() == 0) {
        return mul_out_sparse_scalar(r_, src, value);
    }

    TORCH_CHECK(at::sparse::is_same_density(t, src),
        "add: expected 'self' and 'other' to have same density, but 'self' has ",
        t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions",
        OPS_ERROR(ErrCode::VALUE));

    // We deliberately choose to simply concat the indices and values tensors
    // rather than merging them. This removes the need to synchronously fetch nnz
    // at the end of the operation, at the cost of having a non-coalesced result.
    // This trade-off is preferable for the common use-case of gradient accumulation.
    at::Tensor t_indices_ = t._indices();
    at::Tensor s_indices_ = src._indices();

    at::Tensor t_values_ = t._values().to(common_dtype);
    at::Tensor s_values_ = src._values().to(common_dtype);

    s_values_ = s_values_.mul(value);

    at::Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
    at::Tensor r_values_ = at::cat({t_values_, s_values_}, 0);

    if (r_.scalar_type() != common_dtype) {
        at::sparse::SparseTensor promoted = at::empty({0}, r_.options().dtype(common_dtype));
        promoted.resize_as_(src);
        at::sparse::alias_into_sparse(promoted, r_indices_, r_values_);
        // performs the addition under the common dtype.
        promoted = promoted.coalesce();
        r_values_ = promoted._values().to(r_.scalar_type());
        r_indices_ = promoted._indices();
    } else {
        r_.resize_as_(src);
    }

    alias_into_sparse(r_, r_indices_, r_values_);

    if (r_._nnz() > r_.numel()) {
        auto c = r_.coalesce();
        alias_into_sparse(r_, c._indices(), c._values());
    }

    return r_;
}

}
