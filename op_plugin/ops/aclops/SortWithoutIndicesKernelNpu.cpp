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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& sort_without_indices_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    bool descending)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SortV2")
        .Input(self)
        .Output(result)
        .Attr("axis", dim)
        .Attr("descending", descending)
        .Run();

    return result;
}
} // namespace

at::Tensor& npu_sort_v2_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        self);

    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    int64_t last_dim = op_plugin::utils::make_warp_dim(-1, self.dim());
    if (dim != last_dim) {
        c10::SmallVector<int64_t, SIZE> perm;
        for (int64_t i = 0; i < self.dim(); i++) {
            perm.emplace_back(i);
        }
        std::swap(perm[dim], perm[last_dim]);
        at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);

        auto output_size = op_infer::transpose_npu_output_size(result, perm);
        at::Tensor transpose_result = npu_preparation::apply_tensor(result, output_size);

        sort_without_indices_out_nocheck(transpose_result, transpose_self, last_dim, descending);
        acl_op::npu_transpose_out(transpose_result, perm, true, result);
    } else {
        if (!npu_utils::check_match(&result)) {
            at::Tensor contiguous_result = npu_utils::format_contiguous(result);
            sort_without_indices_out_nocheck(contiguous_result, self, dim, descending);
            npu_utils::format_fresh_view(result, contiguous_result);
        } else {
            sort_without_indices_out_nocheck(result, self, dim, descending);
        }
    }

    return result;
}

at::Tensor npu_sort_v2(
    const at::Tensor& self,
    int64_t dim,
    bool descending)
{
    at::Tensor result = npu_preparation::apply_tensor(self);

    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    int64_t last_dim = op_plugin::utils::make_warp_dim(-1, self.dim());
    if (dim != last_dim) {
        c10::SmallVector<int64_t, SIZE> perm;
        for (int64_t i = 0; i < self.dim(); i++) {
            perm.emplace_back(i);
        }
        std::swap(perm[dim], perm[last_dim]);
        at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);

        auto output_size = op_infer::transpose_npu_output_size(result, perm);
        at::Tensor transpose_result = npu_preparation::apply_tensor(result, output_size);

        sort_without_indices_out_nocheck(transpose_result, transpose_self, last_dim, descending);
        acl_op::npu_transpose_out(transpose_result, perm, true, result);
    } else {
        sort_without_indices_out_nocheck(result, self, dim, descending);
    }

    return result;
}
} // namespace acl_op
