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

bool equal(const at::Tensor& self, const at::Tensor& other)
{
    if (self.sizes() != other.sizes()) {
        return false;
    }

    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "Expected object of scalar type ",
        self.scalar_type(),
        ", but got ",
        other.scalar_type(),
        " for argument #2 'other' in call to equal_npu"
        + OPS_ERROR(ErrCode::TYPE));

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        {1},
        self.options().dtype(at::kBool),
        ACL_FORMAT_ND);

    at_npu::native::OpCommand cmd;
    cmd.Name("TensorEqual")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();

    return result.item().to<bool>();
}
} // namespace acl_op
