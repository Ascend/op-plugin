// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
        std::tuple<at::Tensor &, at::Tensor &, at::Tensor &> unique_dim_out_nocheck(
            at::Tensor &output,
            at::Tensor &idx,
            at::Tensor &inverse_indices,
            at::Tensor &counts,
            const at::Tensor &self,
            const int64_t dim,
            const bool sorted,
            const bool return_inverse)
        {
            c10::SmallVector<int64_t, N> output_sync_idx = {0, 2, 3};
            at_npu::native::OpCommand cmd;
            cmd.Sync(output_sync_idx)
                .Name("UniqueWithCountsExt2")
                .Input(self)
                .Input(dim)
                .Output(output)
                .Output(idx)
                .Output(counts)
                .Output(inverse_indices)
                .Attr("sorted", sorted)
                .Attr("return_inverse", return_inverse);
            cmd.Run();
            return std::tie(output, inverse_indices, counts);
        }
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim(
        const at::Tensor &self,
        int64_t dim,
        bool sorted,
        bool return_inverse,
        bool return_counts)
    {
        auto sizes = self.sizes().vec();
        // check how many zero dimensions exist
        auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);
        // tensor is not well formed as it has 0 sized dimensions
        if (self.size(dim) == 0) {
            TORCH_CHECK(num_zero_dims == 1,
                "Number of zero sized dimensions is more than one, so unique cannot be applied."
                + OPS_ERROR(ErrCode::PARAM));
            at::Tensor output = npu_preparation::apply_tensor_with_format(sizes, self.options(), ACL_FORMAT_ND);
            at::Tensor inverse_indices = npu_preparation::apply_tensor_with_format(
                {0}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
            at::Tensor counts = npu_preparation::apply_tensor_with_format(
                {0}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
            return std::tie(output, inverse_indices, counts);
        }
        TORCH_CHECK(num_zero_dims == 0,
            "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied."
            + OPS_ERROR(ErrCode::PARAM));

        at::Tensor output = npu_preparation::apply_tensor(self);
        at::Tensor idx = npu_preparation::apply_tensor_with_format(
            self.size(dim), self.options().dtype(at::kLong), ACL_FORMAT_ND);
        at::Tensor inverse_indices = npu_preparation::apply_tensor_with_format(
            self.size(dim), self.options().dtype(at::kLong), ACL_FORMAT_ND);
        at::Tensor counts = npu_preparation::apply_tensor_with_format(
            self.size(dim), self.options().dtype(at::kLong), ACL_FORMAT_ND);
        unique_dim_out_nocheck(output, idx, inverse_indices, counts, self, dim, sorted, return_inverse);
        return std::tie(output, inverse_indices, counts);
    }

} // namespace acl_op
