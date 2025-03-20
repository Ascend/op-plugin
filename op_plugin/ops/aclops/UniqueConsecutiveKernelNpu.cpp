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
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_consecutive_out_nocheck(
    at::Tensor& output,
    at::Tensor& inverse_indices,
    at::Tensor& counts,
    const at::Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    c10::optional<int64_t> dim)
{
    at::Tensor self_copy = self;
    if (self.scalar_type() == at::ScalarType::Half) {
        self_copy = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
        output = at_npu::native::custom_ops::npu_dtype_cast(output, at::ScalarType::Float);
    }
    c10::SmallVector<int64_t, N> output_sync_idx = {0, 2};
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("UniqueConsecutive")
        .Input(self_copy)
        .Output(output)
        .Output(inverse_indices)
        .Output(counts)
        .Attr("return_idx", return_inverse)
        .Attr("return_counts", return_counts);
    if (dim.has_value()) {
        cmd.Attr("axis", dim.value());
    }
    cmd.Run();
    if (self.scalar_type() == at::ScalarType::Half) {
        output = at_npu::native::custom_ops::npu_dtype_cast(output, at::ScalarType::Half);
    }
    return std::tie(output, inverse_indices, counts);
}
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> unique_consecutive_out_npu(
    at::Tensor& output,
    at::Tensor& inverse_indices,
    at::Tensor& counts,
    const at::Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    c10::optional<int64_t> dim)
{
    if (dim.has_value()) {
        npu_preparation::CheckOut({self}, output, self);
        npu_preparation::CheckOut({self}, inverse_indices, ACL_FORMAT_ND, self.scalar_type(), self.size(dim.value()));
        npu_preparation::CheckOut({self}, counts, ACL_FORMAT_ND, self.scalar_type(), self.size(dim.value()));
    } else {
        npu_preparation::CheckOut({self}, output, self, {self.numel()});
        npu_preparation::CheckOut({self}, inverse_indices, ACL_FORMAT_ND, self.scalar_type(), self.sizes());
        npu_preparation::CheckOut({self}, counts, ACL_FORMAT_ND, self.scalar_type(), self.numel());
    }

    bool output_match = npu_utils::check_match(&output);
    bool indices_match = npu_utils::check_match(&inverse_indices);
    bool counts_match = npu_utils::check_match(&counts);
    if (output_match && indices_match && counts_match) {
        unique_consecutive_out_nocheck(
            output, inverse_indices, counts, self, return_inverse, return_counts, dim);
    } else {
        at::Tensor contig_output = output_match ? output : npu_utils::format_contiguous(output);
        at::Tensor contig_indices = indices_match ? inverse_indices : npu_utils::format_contiguous(inverse_indices);
        at::Tensor contig_counts = counts_match ? counts : npu_utils::format_contiguous(counts);
        unique_consecutive_out_nocheck(
            contig_output, contig_indices, contig_counts, self, return_inverse, return_counts, dim);
        if (!output_match) {
            npu_utils::format_fresh_view(output, contig_output);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(inverse_indices, contig_indices);
        }
        if (!counts_match) {
            npu_utils::format_fresh_view(counts, contig_counts);
        }
    }

    return std::tie(output, inverse_indices, counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive(
    const at::Tensor& self,
    bool return_inverse,
    bool return_counts,
    c10::optional<int64_t> dim)
{
    at::Tensor output = (dim.has_value()) ?
        npu_preparation::apply_tensor(self) : npu_preparation::apply_tensor(self, {self.numel()});
    at::Tensor inverse_indices = (dim.has_value()) ?
        npu_preparation::apply_tensor_with_format(self.size(dim.value()), self.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);
    at::Tensor counts = (dim.has_value()) ?
        npu_preparation::apply_tensor_with_format(self.size(dim.value()), self.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format({self.numel()}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
    unique_consecutive_out_nocheck(output, inverse_indices, counts, self, return_inverse, return_counts, dim);
    return std::tie(output, inverse_indices, counts);
}

} // namespace acl_op
