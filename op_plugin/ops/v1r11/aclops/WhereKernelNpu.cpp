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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor _s_where(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    at::Tensor result = npu_preparation::apply_tensor(self);

    at_npu::native::OpCommand cmd;
    cmd.Name("Select").Input(condition).Input(self).Input(other).Output(result).Run();

    return result;
}

at::Tensor where(const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other)
{
    TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
                "expected condition, x and y to be on the same device, but condition is on ", condition.device(),
                " and x and y are on ", self.device(), " and ", other.device(), " respectively");
    if (condition.scalar_type() != at::ScalarType::Byte && condition.scalar_type() != at::ScalarType::Bool) {
        AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ", toString(condition.scalar_type()));
    }
    at::Tensor b_condition;
    at::Tensor b_self;
    at::Tensor b_other;
    std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
    return at::_s_where(b_condition, b_self, b_other);
}

} // namespace acl_op
