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
at::Tensor& mean_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& out)
{
    return acl_op::mean_out(self, dimnames_to_positions(self, dim), keepdim, dtype, out);
}

at::Tensor mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    return acl_op::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype)
{
    return acl_op::mean(self, c10::SmallVector<int64_t, N> {}, false, dtype);
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& mean_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& out)
{
    return mean_out_common_nocheck(self, dim, keepdim, dtype, out);
}

at::Tensor mean(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    return mean_common_nocheck(self, dim, keepdim, dtype);
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor& mean_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& out)
{
    return mean_out_common_nocheck(self, dim.value(), keepdim, dtype, out);
}

at::Tensor mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    return mean_common_nocheck(self, dim.value(), keepdim, dtype);
}
#endif
} // namespace acl_op
