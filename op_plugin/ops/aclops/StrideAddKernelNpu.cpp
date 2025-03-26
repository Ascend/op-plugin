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

namespace {
at::Tensor& stride_add_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::Scalar offset1,
    c10::Scalar offset2,
    c10::Scalar c1_len)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("StrideAdd")
        .Input(self, "x1")
        .Input(other, "x2")
        .Output(result, "y")
        .Attr("x1_c1_offset", static_cast<int64_t>(offset1.toInt()))
        .Attr("x2_c1_offset", static_cast<int64_t>(offset2.toInt()))
        .Attr("c1_len", static_cast<int64_t>(c1_len.toInt()))
        .Run();
    return result;
}
} // namespace

at::Tensor npu_stride_add(
    const at::Tensor& self,
    const at::Tensor& other,
    const c10::Scalar& offset1,
    const c10::Scalar& offset2,
    const c10::Scalar& c1_len)
{
    auto output_size = op_infer::infersize_stride_add(self.sizes(), other.sizes());
    output_size[1] = c1_len.toInt() * 16;
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    stride_add_out_npu_nocheck(result, self, other, offset1, offset2, c1_len);
    return result;
}
}  // namespace acl_op
