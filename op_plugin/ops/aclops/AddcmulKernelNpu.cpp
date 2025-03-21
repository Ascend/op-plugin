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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& addcmul_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Addcmul")
        .Input(self)
        .Input(tensor1)
        .Input(tensor2)
        .Input(value, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& addcmul_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& result)
{
    auto input_size = op_infer::broadcast_ops_npu_output_size(self, tensor1);
    auto output_size = op_infer::broadcast_ops_npu_output_size(input_size, tensor2.sizes());

    npu_preparation::CheckOut(
        {self, tensor1, tensor2},
        result,
        self,
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        addcmul_out_npu_nocheck(contiguous_result, self, tensor1, tensor2, value);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);
    }
    return result;
}

at::Tensor addcmul(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value)
{
    auto input_size = op_infer::broadcast_ops_npu_output_size(self, tensor1);
    auto output_size = op_infer::broadcast_ops_npu_output_size(input_size, tensor2.sizes());
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);
    return result;
}

at::Tensor& addcmul_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value)
{
    return acl_op::addcmul_out(self, tensor1, tensor2, value, self);
}
}  // namespace acl_op
