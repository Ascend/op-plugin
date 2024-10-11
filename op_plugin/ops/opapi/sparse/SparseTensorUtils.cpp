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

#include "SparseTensorUtils.h"

#include <ATen/native/DispatchStub.h>

namespace sparse {
at::Tensor flatten_indices_npu_kernel(const at::Tensor& indices, c10::IntArrayRef size)
{
    std::vector<int64_t> flatten_size(size.size(), 1);
    for (size_t i = size.size() - 1; i > 0; i--) {
        flatten_size[i - 1] = flatten_size[i] * size[i];
    }
    auto tensor_temp = torch::zeros({indices.size(1)}, indices.options());
    for (size_t i = 0; i < size.size(); i++) {
        tensor_temp += indices[i] * flatten_size[i];
    }
    return tensor_temp;
}

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const at::Tensor& value)
{
    AT_ASSERT(r.is_sparse());
    AT_ASSERT(t.is_sparse());
    AT_ASSERT(value.dim() == 0);

    at::Tensor value_;
    if (value.is_sparse()) {
        if (value._nnz() == 0) {
            r.resize_as_(t);
            return r.zero_();
        }
        value_ = value.values();
    } else {
        value_ = value;
    }
    // With broadcasting in action, value_ may be a 1-D tensor as long
    // as its shape is (1,).
    AT_ASSERT(value_.numel() == 1);

    if (is_same_tensor(r, t)) {
        r._values().mul_(value_);
    } else {
        r.resize_as_(t);
        auto indices = r._indices();
        indices.resize_as_(t._indices());
        indices.copy_(t._indices());
        at::Tensor r_values = r._values();
        at::mul_out(r_values, t._values(), value_);
        get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
        r._coalesced_(t.is_coalesced());
    }
    return r;
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const at::Scalar& value)
{
    return mul_out_sparse_zerodim(r, t, at::native::wrapped_scalar_tensor(value));
}

}

namespace at {
namespace native {
using flatten_indices_fn = at::Tensor (*)(const at::Tensor& indices, at::IntArrayRef size);
DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub);
REGISTER_PRIVATEUSE1_DISPATCH(flatten_indices_stub, &::sparse::flatten_indices_npu_kernel);
}
}
