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

namespace {
#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, V2R1)
std::tuple<at::Tensor&, at::Tensor&> _unique_out_npu(
    at::Tensor& y,
    at::Tensor& y_inverse,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse)
{
    c10::SmallVector<int64_t, N> output_sync_idx = {0, 1, 2};
    at::Tensor y_counts = npu_preparation::apply_tensor_with_format({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("UniqueWithCountsAndSorting")
        .Input(self)
        .Output(y)
        .Output(y_inverse)
        .Output(y_counts)
        .Attr("sorted", sorted)
        .Attr("return_inverse", return_inverse)
        .Attr("return_counts", false)
        .Run();

    return std::tuple<at::Tensor&, at::Tensor&>(y, y_inverse);
}
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
std::tuple<at::Tensor&, at::Tensor&> _unique_out_npu(
    at::Tensor& y,
    at::Tensor& y_inverse,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse)
{
    c10::SmallVector<int64_t, N> output_sync_idx = {0, 1, 2};
    at::Tensor y_counts = npu_preparation::apply_tensor_with_format({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND);
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("UniqueWithCountsAndSorting")
        .Input(self)
        .Output(y)
        .Output(y_inverse)
        .Output(y_counts)
        .Attr("sorted", true)
        .Attr("return_inverse", return_inverse)
        .Attr("return_counts", false)
        .Run();

    return std::tuple<at::Tensor&, at::Tensor&>(y, y_inverse);
}
#endif
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor, at::Tensor> _unique(
    const at::Tensor& self_op,
    bool sorted,
    bool return_inverse)
{
    const at::Tensor self = self_op.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(self_op, at::kFloat) : self_op;
    if (self.numel() == 0) {
        at::Tensor result = npu_preparation::apply_tensor(self, {0});
        at::Tensor y_inverse = npu_preparation::apply_tensor({0}, self.options().dtype(at::kLong), self);
        return std::tie(result, y_inverse);
    }
    at::Tensor y = npu_preparation::apply_tensor(self, self.numel());
    at::Tensor y_inverse = !return_inverse ?
        npu_preparation::apply_tensor_with_format({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);

    _unique_out_npu(y, y_inverse, self, sorted, return_inverse);
    if (self_op.scalar_type() == at::kHalf) {
        y = at_npu::native::custom_ops::npu_dtype_cast(y, at::kHalf);
    }
    return std::tuple<at::Tensor, at::Tensor>(y, y_inverse);
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
std::tuple<at::Tensor, at::Tensor> _unique(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse)
{
    if (self.numel() == 0) {
        at::Tensor result = npu_preparation::apply_tensor(self, {0});
        at::Tensor y_inverse = npu_preparation::apply_tensor({0}, self.options().dtype(at::kLong), self);
        return std::tie(result, y_inverse);
    }
    at::Tensor y = npu_preparation::apply_tensor(self, self.numel());
    at::Tensor y_inverse = !return_inverse ?
        npu_preparation::apply_tensor_with_format({1}, self.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format(self.sizes(), self.options().dtype(at::kLong), ACL_FORMAT_ND);

    _unique_out_npu(y, y_inverse, self, sorted, return_inverse);
    return std::tuple<at::Tensor, at::Tensor>(y, y_inverse);
}
#endif
}  // namespace acl_op
