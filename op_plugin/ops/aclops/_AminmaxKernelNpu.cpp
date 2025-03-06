// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor& self)
{
    auto min = acl_op::min(self);
    auto max = acl_op::max(self);
    return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> _aminmax(
    const at::Tensor& self,
    const int64_t dim,
    const bool keepdim)
{
    auto min = acl_op::min(self, {dim}, keepdim);
    auto max = acl_op::max(self, {dim}, keepdim);
    return std::tie(std::get<0>(min), std::get<0>(max));
}

std::tuple<at::Tensor&, at::Tensor&> aminmax_out(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& min,
    at::Tensor& max)
{
    if (dim.has_value()) {
        max = acl_op::amax_out(self, dim.value(), keepdim, max);
        min = acl_op::amin_out(self, dim.value(), keepdim, min);
    } else {
        at::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
        max = acl_op::amax_out(self, dims, keepdim, max);
        min = acl_op::amin_out(self, dims, keepdim, min);
    }
    return std::tie(min, max);
}
} // namespace acl_op
