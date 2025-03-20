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
at::Tensor& searchsorted_out_nocheck(
    at::Tensor& result,
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right)
{
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    at_npu::native::OpCommand cmd;
    cmd.Name("SearchSorted")
        .Input(sorted_sequence)
        .Input(self)
        .Attr("dtype", scalar_type)
        .Attr("right", right)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& searchsorted_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt,
    at::Tensor& result)
{
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    npu_preparation::CheckOut(
        {sorted_sequence, self},
        result,
        npu_preparation::get_tensor_npu_format(self),
        scalar_type,
        self.sizes());
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        searchsorted_out_nocheck(contiguous_result, sorted_sequence, self, out_int32, right);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        searchsorted_out_nocheck(result, sorted_sequence, self, out_int32, right);
    }

    return result;
}

at::Tensor searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt)
{
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    at::Tensor result = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(scalar_type), self);
    searchsorted_out_nocheck(result, sorted_sequence, self, out_int32, right);
    return result;
}

at::Tensor searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt)
{
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    at::Tensor self_op = npu_preparation::copy_scalar_to_device(self, sorted_sequence.scalar_type());
    self_op = self_op.unsqueeze(0);
    at::Tensor result = npu_preparation::apply_tensor({}, sorted_sequence.options().dtype(scalar_type), sorted_sequence);
    searchsorted_out_nocheck(result, sorted_sequence, self_op, out_int32, right);
    return result;
}
} // namespace acl_op
