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
using npu_compile_type = at_npu::native::CompileType;

at::Tensor one_hot(const at::Tensor& self, int64_t num_classes) {
    at::Scalar on_value = 1;
    at::Scalar off_value = 0;
    int64_t axis = -1;
    int64_t depth;
    auto self_temp = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);

    if (self.numel() == 0) {
        TORCH_CHECK(num_classes > 0, "Can not infer total number of classes from empty tensor."
            + OPS_ERROR(ErrCode::PARAM));
        depth = num_classes;
    }

    if (num_classes == -1) {
        depth = self_temp.max().item().toLong() + 1;
    } else {
        depth = num_classes;
    }

    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.emplace_back(depth);
    at::Tensor result = npu_preparation::apply_tensor(output_size, self.options(), self);
    at::Scalar depth_copy = depth;
    at_npu::native::OpCommand cmd;
    cmd.Name("OneHot")
        .Input(self)
        .Input(depth_copy, at::kInt, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(on_value, self.scalar_type())
        .Input(off_value, self.scalar_type())
        .Output(result)
        .Attr("axis", axis)
        .Run();
    return result;
}
} // namespace acl_op
