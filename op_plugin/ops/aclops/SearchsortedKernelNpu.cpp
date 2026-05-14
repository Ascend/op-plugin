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
#include "op_plugin/utils/SearchsortedSideUtil.h"
#include "op_plugin/utils/SearchsortedValidateUtil.h"
#include "op_plugin/utils/SearchsortedWarnUtil.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor &searchsorted_out_nocheck(
    at::Tensor &result, const at::Tensor &sorted_sequence, const at::Tensor &self, bool out_int32, bool right) {
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

at::Tensor &searchsorted_out(const at::Tensor &sorted_sequence, const at::Tensor &self, bool out_int32, bool right,
    const c10::optional<c10::string_view> side_opt, const c10::optional<at::Tensor> &sorter_opt, at::Tensor &result) {
    (void)op_plugin::searchsorted_validate_tensor_out_op(
        sorted_sequence, self, result, out_int32, right, side_opt, sorter_opt);
    (void)op_plugin::warn_if_searchsorted_inputs_noncontiguous(sorted_sequence, self, sorter_opt);
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    npu_preparation::CheckOut(
        {sorted_sequence, self}, result, npu_preparation::get_tensor_npu_format(self), scalar_type, self.sizes());
    const bool right_eff = op_plugin::resolve_searchsorted_effective_right(right, side_opt);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        searchsorted_out_nocheck(contiguous_result, sorted_sequence, self, out_int32, right_eff);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        searchsorted_out_nocheck(result, sorted_sequence, self, out_int32, right_eff);
    }

    return result;
}

at::Tensor searchsorted(const at::Tensor &sorted_sequence, const at::Tensor &self, bool out_int32, bool right,
    const c10::optional<c10::string_view> side_opt, const c10::optional<at::Tensor> &sorter_opt) {
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    at::Tensor result = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(scalar_type), self);
    (void)op_plugin::searchsorted_validate_tensor_out_op(
        sorted_sequence, self, result, out_int32, right, side_opt, sorter_opt);
    (void)op_plugin::warn_if_searchsorted_inputs_noncontiguous(sorted_sequence, self, sorter_opt);
    const bool right_eff = op_plugin::resolve_searchsorted_effective_right(right, side_opt);
    searchsorted_out_nocheck(result, sorted_sequence, self, out_int32, right_eff);
    return result;
}

at::Tensor searchsorted(const at::Tensor &sorted_sequence, const at::Scalar &self, bool out_int32, bool right,
    const c10::optional<c10::string_view> side_opt, const c10::optional<at::Tensor> &sorter_opt) {
    at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
    (void)op_plugin::searchsorted_validate_scalar_op(sorted_sequence, self, out_int32, right, side_opt, sorter_opt);
    (void)op_plugin::warn_if_searchsorted_scalar_inputs_noncontiguous(sorted_sequence, sorter_opt);
    at::Tensor self_op = npu_preparation::copy_scalar_to_device(self, sorted_sequence.scalar_type());
    self_op = self_op.unsqueeze(0);
    at::Tensor result =
        npu_preparation::apply_tensor({}, sorted_sequence.options().dtype(scalar_type), sorted_sequence);
    const bool right_eff = op_plugin::resolve_searchsorted_effective_right(right, side_opt);
    searchsorted_out_nocheck(result, sorted_sequence, self_op, out_int32, right_eff);
    return result;
}
} // namespace acl_op
