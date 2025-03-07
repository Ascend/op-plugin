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
#include <ATen/native/NonSymbolicBC.h>

#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "op_plugin/SparseOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#include "SparseTensorUtils.h"

namespace sparse {

using namespace at::sparse;

SparseTensor _coalesce_sparse(const SparseTensor& self)
{
    int64_t nnz = self._nnz();
    TORCH_CHECK(!self.is_coalesced(), OPS_ERROR(ErrCode::VALUE));
    if (nnz < 2) {
        SparseTensor dst = self.clone();
        dst._coalesced_(true);
        return dst;
    }

    at::Tensor values = self._values();
    at::Tensor indices = self._indices();
    at::Tensor indices_1d = at::sparse::flatten_indices(indices, self.sizes(), true);
    auto unique_indices_info = at::_unique2(indices_1d, true, true);
    at::Tensor unique_len = std::get<0>(unique_indices_info).to(at::kInt);
    auto new_nnz = unique_len.sizes()[0];
    auto new_values_size = values.sizes().vec();
    new_values_size[0] = new_nnz;
    at::Tensor new_indices_t = at::zeros(
        {new_nnz, indices.sizes()[0]},
        indices.options().dtype(at::kInt));
    at::Tensor indices_t = at_npu::native::NpuUtils::format_contiguous(indices.transpose(0, 1)).to(at::kInt);
    at::Tensor unique_indices = std::get<1>(unique_indices_info).to(at::kInt);
    if (values.scalar_type() == at::kHalf || (values.scalar_type() == at::kBFloat16)) {
        at::Tensor values_f = values.to(at::kFloat);
        at::Tensor new_values_f = at::zeros(new_values_size, values.options()).to(at::kFloat);
        EXEC_NPU_CMD(
            aclnnCoalesceSparse,
            unique_len,
            unique_indices,
            indices_t,
            values_f,
            new_indices_t,
            new_values_f);
        at::Tensor new_indices = new_indices_t.transpose(0, 1).to(at::kLong);
        if (values.scalar_type() == at::kHalf) {
            at::Tensor new_values = new_values_f.to(at::kHalf);
            SparseTensor dst = ::at::native::_sparse_coo_tensor_unsafe(new_indices,
                new_values, self.sizes())._coalesced_(true);
            return dst;
        } else {
            at::Tensor new_values = new_values_f.to(at::kBFloat16);
            SparseTensor dst = ::at::native::_sparse_coo_tensor_unsafe(new_indices,
                new_values, self.sizes())._coalesced_(true);
            return dst;
        }
    } else {
        at::Tensor new_values = at::zeros(new_values_size, values.options());
        EXEC_NPU_CMD(
            aclnnCoalesceSparse,
            unique_len,
            unique_indices,
            indices_t,
            values,
            new_indices_t,
            new_values);
        at::Tensor new_indices = new_indices_t.transpose(0, 1).to(at::kLong);
        SparseTensor dst = ::at::native::_sparse_coo_tensor_unsafe(new_indices,
            new_values, self.sizes())._coalesced_(true);
        return dst;
    }
}

} // namespace at::native
