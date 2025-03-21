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

namespace {
at::Tensor& one_hot_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t axis,
    int64_t depth,
    at::Scalar on_value,
    at::Scalar off_value)
{
    at::Tensor self_copy = at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt);
    at::Tensor on_tmp = npu_preparation::apply_tensor(
        {1},
        self_copy.options().dtype(at::ScalarType::Float),
        self_copy);
    acl_op::fill_(on_tmp, on_value);

    at::Tensor off_tmp = npu_preparation::apply_tensor(
        {1},
        self_copy.options().dtype(at::ScalarType::Float),
        self_copy);
    acl_op::fill_(off_tmp, off_value);

    at_npu::native::OpCommand cmd;
    cmd.Name("OneHotD")
        .Input(self_copy)
        .Input(on_tmp)
        .Input(off_tmp)
        .Output(result)
        .Attr("axis", axis)
        .Attr("depth", depth)
        .Run();
    return result;
}
} // namespace

at::Tensor npu_one_hot(
    const at::Tensor& self,
    int64_t num_classes,
    int64_t depth,
    const at::Scalar& on_value,
    const at::Scalar& off_value)
{
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.emplace_back(depth);

    at::Tensor result = npu_preparation::apply_tensor(
        output_size,
        self.options().dtype(at::ScalarType::Float),
        self);
    one_hot_out_npu(result, self, num_classes, depth, on_value, off_value);
    return result;
}
} // namespace acl_op
