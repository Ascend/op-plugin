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
#include "op_plugin/OpInterface.h"

namespace sparse {
at::Tensor flatten_indices_npu_kernel(const at::Tensor& indices, c10::IntArrayRef size)
{
    std::vector<int64_t> flatten_size(size.size(), 1);
    for (int i = size.size() - 1; i > 0; i--) {
        flatten_size[i - 1] = flatten_size[i - 1] * size[i];
    }
    auto tensor_temp = torch::tensor(flatten_size, indices.options().dtype(at::kInt));
    tensor_temp = torch::unsqueeze(tensor_temp, 0);
    return torch::squeeze(at::matmul(tensor_temp, indices.to(at::kInt)), 0);
}

at::Tensor flatten_indices(const at::Tensor& indices, c10::IntArrayRef full_size)
{
    int64_t sparse_dim = indices.size(0);
    if (sparse_dim == 1) {
        return indices.squeeze(0);
    } else {
        if (!indices.numel()) {
            return at::zeros({indices.size(1)}, indices.options().dtype(at::kLong));
        }
        return flatten_indices_npu_kernel(indices, full_size.slice(0, sparse_dim));
    }
}

at::Tensor& add_out_dense_sparse_npu(
    at::Tensor& r_,
    const at::Tensor& dense,
    const at::sparse::SparseTensor& sparse,
    const at::Scalar& value)
{
    TORCH_CHECK(torch_npu::utils::is_npu(dense), "sparse add: expected 'self' to be a NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(sparse), "sparse add: expected 'other' to be a NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(r_), "sparse add: expected 'out' to be a NPU tensor");

    TORCH_CHECK(dense.sizes().equals(sparse.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
        dense.sizes(), " while other has size ", sparse.sizes(),
        " (FYI: dense-sparse addition does not currently support broadcasting)");

    const int64_t nnz = sparse._nnz();
    if (nnz == 0) {
        r_.resize_as_(dense);
        r_.copy_(dense);
        return r_;
    }

    auto common_dtype = at::result_type(dense, sparse);
    TORCH_CHECK(c10::canCast(common_dtype, r_.scalar_type()), "Can't convert result type ", common_dtype, " to output ", r_.scalar_type());

    at::Tensor r = r_;
    if (r_.scalar_type() != common_dtype) {
        r = at::empty_like(dense, r_.options().dtype(common_dtype));
    }

    at::Tensor dense_buffer = dense.to(common_dtype);
    at::Tensor values = sparse._values().to(common_dtype);

    if (at::sparse::is_same_tensor(r, dense_buffer)) {
        TORCH_CHECK(r_.is_contiguous(), "add: NPU dense-sparse addition with a non-contiguous output tensor does not work");
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

    at::Tensor indices_1D = flatten_indices(indices, sparse.sizes());

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

    TORCH_CHECK(false, "sparse add: NPU sparse-sparse addition does not work yet");
    return r_;
}

}
