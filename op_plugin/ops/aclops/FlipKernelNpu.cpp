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

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims)
{
    if (dims.size() == 0) {
        return self.clone();
    }

    at::Tensor result = npu_preparation::apply_tensor(self);
    at::SmallVector<int64_t, N> dim_vector = op_infer::array_to_small_vector(dims);
    at_npu::native::OpCommand cmd;
    cmd.Name("ReverseV2")
        .Input(self)
        .Input(dim_vector, at::kLong)
        .Output(result)
        .Run();
    return result;
}
} // namespace acl_op
