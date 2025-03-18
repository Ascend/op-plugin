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
at::Tensor& zeros_out(at::IntArrayRef size, at::Tensor& out)
{
    out.resize_(size);
    return out.zero_();
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return zeros_common_nocheck(size, dtype, layout, device, pin_memory);
}

at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return zeros_common_nocheck(size, dtype, layout, device, pin_memory);
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return zeros_common_nocheck(size, dtype, layout, device, pin_memory);
}

at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return acl_op::zeros(size, dtype, layout, device, pin_memory);
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor zeros_symint(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return zeros_common_nocheck(c10::asIntArrayRefUnchecked(size), dtype, layout, device, pin_memory);
}


at::Tensor zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return zeros_common_nocheck(size, dtype, layout, device, pin_memory);
}
#endif
} // namespace acl_op
