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
std::tuple<at::Tensor&, at::Tensor&> sort_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool descending)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Sort")
        .Input(self)
        .Output(values)
        .Output(indices)
        .Attr("axis", dim)
        .Attr("descending", descending)
        .Run();

    return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> sort_out_npu_transpose(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool descending)
{
    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    int64_t last_dim = op_plugin::utils::make_warp_dim(-1, self.dim());
    if (dim != last_dim) {
        at::SmallVector<int64_t, SIZE> perm;
        for (int64_t i = 0; i < self.dim(); i++) {
            perm.emplace_back(i);
        }
        std::swap(perm[dim], perm[last_dim]);

        at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);
        auto output_size = op_infer::transpose_npu_output_size(values, perm);
        at::Tensor transpose_values = npu_preparation::apply_tensor(values, output_size);
        at::Tensor transpose_indices = npu_preparation::apply_tensor(indices, output_size);

        sort_out_npu_nocheck(transpose_values, transpose_indices, transpose_self, last_dim, descending);

        acl_op::npu_transpose_out(transpose_values, perm, true, values);
        acl_op::npu_transpose_out(transpose_indices, perm, true, indices);
    } else {
        sort_out_npu_nocheck(values, indices, self, last_dim, descending);
    }
    return std::tie(values, indices);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> sort_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices)
{
    auto output_size = op_infer::input_same_output_size(self);
    npu_preparation::CheckOut(
        {self},
        values,
        self);
    npu_preparation::CheckOut(
        {self},
        indices,
        ACL_FORMAT_NCHW,
        at::ScalarType::Long,
        output_size);

    at::Tensor indices_cp = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kInt);
    bool values_match = npu_utils::check_match(&values);
    bool indices_match = npu_utils::check_match(&indices_cp);
    if (!(values_match && indices_match)) {
        at::Tensor contiguous_values = values_match ? values : npu_utils::format_contiguous(values);
        at::Tensor contiguous_indices = indices_match ? indices_cp : npu_utils::format_contiguous(indices_cp);
        sort_out_npu_transpose(contiguous_values, contiguous_indices, self, dim, descending);
        if (!values_match) {
            npu_utils::format_fresh_view(values, contiguous_values);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(indices_cp, contiguous_indices);
        }
    } else {
        sort_out_npu_transpose(values, indices_cp, self, dim, descending);
    }

    // indices dtype transform Int64
    indices_cp = at_npu::native::custom_ops::npu_dtype_cast(indices_cp, at::kLong);
    indices.copy_(indices_cp);
    return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> sort_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices)
{
    return acl_op::sort_out(self, dimname_to_position(self, dim), descending, values, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending)
{
    auto output_size = op_infer::input_same_output_size(self);

    at::Tensor values = npu_preparation::apply_tensor(self);
    at::Tensor indices = npu_preparation::apply_tensor_with_format(
        output_size, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);

    sort_out_npu_transpose(values, indices, self, dim, descending);
    // indices dtype transform Int64
    indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kLong);
    return std::tie(values, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending)
{
    return acl_op::sort(self, dimname_to_position(self, dim), descending);
}

} // namespace acl_op
