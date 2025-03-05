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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R0, V2R0)
at::Tensor repeat_interleave(
    const at::Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size)
{
    return repeat_interleave_common_nocheck(self, repeats, dim);
}
#endif

#if VERSION_BETWEEN(V2R1, V2R1)
at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size)
{
    int64_t repeats_val = repeats.guard_int(__FILE__, __LINE__);
    return repeat_interleave_common_nocheck(self, repeats_val, dim);
}
#endif

#if VERSION_BETWEEN(V1R11, V2R1)
at::Tensor repeat_interleave(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size)
{
    return repeat_interleave_common_nocheck(self, repeats, dim);
}
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<c10::SymInt> output_size)
{
    int64_t repeats_val = repeats.guard_int(__FILE__, __LINE__);
    c10::optional<int64_t> _output_size = c10::nullopt;
    if (output_size.has_value()) {
        int64_t output_size_val = output_size.value().guard_int(__FILE__, __LINE__);
        _output_size = c10::optional<int64_t>(output_size_val);
    }
    return repeat_interleave_common_nocheck(self, repeats_val, dim);
}

at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<c10::SymInt> output_size)
{
    c10::optional<int64_t> _output_size = c10::nullopt;
    if (output_size.has_value()) {
        int64_t output_size_val = output_size.value().guard_int(__FILE__, __LINE__);
        _output_size = c10::optional<int64_t>(output_size_val);
    }
    return repeat_interleave_common_nocheck(self, repeats, dim);
}
#endif
} // namespace acl_op
