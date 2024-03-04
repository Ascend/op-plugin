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

at::Tensor &where_out_nocheck(at::Tensor &out, const at::Tensor &condition, const at::Tensor &self,
                              const at::Tensor &other)
{
    at::Tensor self_cp;
    at::Tensor other_cp;
    if (self.dtype() != other.dtype()) {
        auto result_type = at::native::result_type(self, other);
        self_cp = acl_op::npu_dtype_cast(self, result_type);
        other_cp = acl_op::npu_dtype_cast(other, result_type);
    } else {
        self_cp = self;
        other_cp = other;
    }

    TORCH_CHECK(!(condition.scalar_type() != at::ScalarType::Byte && condition.scalar_type() != at::ScalarType::Bool),
                "Expected condition to have ScalarType Byte, but got ScalarType ", toString(condition.scalar_type()),
                OPS_ERROR(ErrCode::TYPE));

    at_npu::native::OpCommand cmd;
    cmd.Name("Select").Input(condition).Input(self_cp).Input(other_cp).Output(out).Run();

    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_expand_outplace(const at::Tensor &to_expand1,
                                                                   const at::Tensor &to_expand2,
                                                                   const at::Tensor &to_expand3, const char *api_name)
{
    for (auto &t : {to_expand1, to_expand2, to_expand3}) {
        if (!t.defined()) {
            TORCH_CHECK(false, api_name, "(...) called with an undefined Tensor", OPS_ERROR(ErrCode::PARAM));
        }
    }

    if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
        return std::make_tuple(to_expand1, to_expand2, to_expand3);
    }

    auto expanded_size12 = op_infer::broadcast_ops_npu_output_size(to_expand1, to_expand2);
    auto expanded_size = op_infer::broadcast_ops_npu_output_size(expanded_size12, to_expand3.sizes());

    return std::make_tuple(to_expand1.expand(expanded_size, true), to_expand2.expand(expanded_size, true),
                           to_expand3.expand(expanded_size, true));
}

} // namespace acl_op
