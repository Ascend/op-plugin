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

at::Tensor npu_sub_sample(
    const at::Tensor& self,
    int64_t per_images,
    double positive_fraction)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    at_npu::native::OpCommand cmd;
    cmd.Name("SubSample")
        .Input(self)
        .Output(result)
        .Attr("batch_size_per_images", per_images)
        .Attr("positive_fraction", (float)positive_fraction)
        .Run();
    return result;
}
} // namespace acl_op
