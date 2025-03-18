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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor isfinite(const at::Tensor& self)
{
    at::Tensor self_ex = self;
    if (npu_preparation::get_tensor_npu_format(self_ex) != ACL_FORMAT_ND) {
        self_ex = at_npu::native::custom_ops::npu_format_cast(self, ACL_FORMAT_ND);
    }
    if (self_ex.scalar_type() == at::ScalarType::Half) {
        self_ex = at_npu::native::custom_ops::npu_dtype_cast(self_ex, at::ScalarType::Float);
    }
    auto output_size = self_ex.sizes();
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, self_ex.options().dtype(at::kBool), ACL_FORMAT_ND);

    at_npu::native::OpCommand cmd;
    cmd.Name("IsFinite")
        .Input(self_ex)
        .Output(result)
        .Run();
    return result;
}
} // namespace acl_op
