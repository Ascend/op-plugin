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
#include <torch/torch.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

namespace sparse {
using namespace at::sparse;

inline at::SparseTensorImpl* get_sparse_impl(const SparseTensor& self)
{
    TORCH_CHECK(self.is_sparse(), "_internal_get_SparseTensorImpl: not a sparse tensor", OPS_ERROR(ErrCode::VALUE));
    return static_cast<at::SparseTensorImpl*>(self.unsafeGetTensorImpl());
}

inline void alias_into_sparse(
    const SparseTensor& self,
    const at::Tensor& indices,
    const at::Tensor& values)
{
    get_sparse_impl(self)->set_indices_and_values_unsafe(indices, values);
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, const at::Scalar& value);

}