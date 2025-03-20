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

at::Tensor& npu_slice_out(
    const at::Tensor& self,
    c10::IntArrayRef offsets,
    c10::IntArrayRef size,
    at::Tensor& result)
{
    c10::SmallVector<int64_t, N> offsetVec = op_infer::array_to_small_vector(offsets);
    c10::SmallVector<int64_t, N> sizeVec = op_infer::array_to_small_vector(size);
    at_npu::native::OpCommand cmd;
    cmd.Name("Slice")
        .Input(self)
        .Input(offsetVec)
        .Input(sizeVec)
        .Output(result)
        .Run();
    return result;
}

at::Tensor npu_slice(const at::Tensor& self, c10::IntArrayRef offsets, c10::IntArrayRef size)
{
    c10::SmallVector<int64_t, SIZE> output_size = op_plugin::utils::convert_array_to_vector(size);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);

    acl_op::npu_slice_out(self, offsets, size, result);

    return result;
}

} // namespace acl_op
