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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

void index_copy_npu_par_check(const int64_t dim, const at::Tensor& index,
                              const at::Tensor& source, const at::Tensor& result)
{
    int64_t new_dim = at::maybe_wrap_dim(dim, result.dim());
    TORCH_CHECK_INDEX(index.dim() < 2, "index_copy_()", ": Index should have dimension 1 or 0 (got ", index.dim(), ")",
        OPS_ERROR(ErrCode::VALUE));

    int64_t num_indices = index.numel();
    TORCH_CHECK_INDEX(!(source.dim() == 0 && num_indices != 1),
        "index_copy_()", ": When source is scalar, index should have one element (got ", num_indices, ")",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK_INDEX(!((source.dim() != result.dim()) && (source.dim() != 0 && result.dim() != 0)),
        "index_copy_()", ": When source and destination are not scalars, "
        "their dimensionality must match. Source dimensionality (",
        source.dim(), "), destination dimensionality (", result.dim(), ")", OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK_INDEX(index.scalar_type() == at::ScalarType::Long, "index_copy_()", ": Expected LongTensor for index",
        OPS_ERROR(ErrCode::TYPE));

    // Check that source and destination slices have the same size
    auto self_sliced_sizes = result.sizes().vec();
    int64_t boundary_index = 0;
    if (self_sliced_sizes.size() > 0) {
        boundary_index = self_sliced_sizes[new_dim] - 1;
        self_sliced_sizes.erase(self_sliced_sizes.begin() + new_dim);
    }
    auto source_sliced_sizes = source.sizes().vec();
    if (source_sliced_sizes.size() > 0) {
        source_sliced_sizes.erase(source_sliced_sizes.begin() + new_dim);
    }

    TORCH_CHECK(!(self_sliced_sizes.size() != source_sliced_sizes.size() ||
        !std::equal(self_sliced_sizes.begin(), self_sliced_sizes.end(), source_sliced_sizes.begin())),
        "index_copy_()", ": Source/destination tensor must have same slice shapes.\n",
        "Destination slice shape: ", self_sliced_sizes, " at dimension ", new_dim,
        " and source slice shape: ", source_sliced_sizes, " at dimension 0.", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK_INDEX(source.dim() == 0 || num_indices == source.size(new_dim),
        "index_copy_()", ": Number of indices (", num_indices,
        ") should be equal to source.size(newDim) (", source.size(new_dim), ")", OPS_ERROR(ErrCode::VALUE));
  
    for (int64_t i = 0; i < num_indices; i++) {
        int64_t specifical_index = index.dim() == 0 ? index.item<int64_t>() : index[i].item<int64_t>();
        TORCH_CHECK_INDEX(specifical_index <= boundary_index, "index_copy_()", ": index ", specifical_index,
            " is out of bounds for dimension ", boundary_index, " with size ", boundary_index + 1,
            OPS_ERROR(ErrCode::VALUE));
    }
}
} // namespace acl_op
