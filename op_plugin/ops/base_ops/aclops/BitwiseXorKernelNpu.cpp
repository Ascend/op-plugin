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
at::Tensor &bitwise_xor_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("BitwiseXor").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &bitwise_xor_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    if (npu_preparation::IsCPUScalar(other)) {
        acl_op::bitwise_xor_out(self, other.item(), result);
    } else if (npu_preparation::IsCPUScalar(self)) {
        acl_op::bitwise_xor_out(other, self.item(), result);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("BitwiseXor").Expect(unified_result).Input(self).Input(other).Output(result).Run();
    }
    return result;
}
} // namespace

at::Tensor &bitwise_xor_out(const at::Tensor &self, const at::Scalar &other, at::Tensor &result)
{
    npu_preparation::CheckOut({self}, result, self);
    at::Tensor self_input =
        (self.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt) : self;
    at::Tensor result_cp =
        (result.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(result, at::kInt) : result;
    if (!npu_utils::check_match(&result_cp)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
        bitwise_xor_out_npu_nocheck(contiguous_result, self_input, other);
        npu_utils::format_fresh_view(result_cp, contiguous_result);
    } else {
        bitwise_xor_out_npu_nocheck(result_cp, self_input, other);
    }
    if (self.dtype() == at::kBool) {
        result_cp = at_npu::native::custom_ops::npu_dtype_cast(result_cp, at::kBool);
        result.copy_(result_cp);
    }
    return result;
}

at::Tensor &bitwise_xor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor output_tensor = is_self_wrapped ? other : self;

    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self, other}, result, output_tensor, output_size);
    at::Tensor self_input =
        (self.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt) : self;
    at::Tensor other_input =
        (other.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(other, at::kInt) : other;
    at::Tensor result_cp =
        (result.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(result, at::kInt) : result;
    if (!npu_utils::check_match(&result_cp)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
        bitwise_xor_out_npu_nocheck(contiguous_result, self_input, other_input);
        npu_utils::format_fresh_view(result_cp, contiguous_result);
    } else {
        bitwise_xor_out_npu_nocheck(result_cp, self_input, other_input);
    }
    if (self.dtype() == at::kBool) {
        result_cp = at_npu::native::custom_ops::npu_dtype_cast(result_cp, at::kBool);
        result.copy_(result_cp);
    }
    return result;
}

at::Tensor bitwise_xor(const at::Tensor &self, const at::Tensor &other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor output_tensor = is_self_wrapped ? other : self;

    at::Tensor self_input =
        (self.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt) : self;
    at::Tensor other_input =
        (other.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(other, at::kInt) : other;
    at::Tensor result =
        output_tensor.dtype() == at::kBool ?
            npu_preparation::apply_tensor(output_size, output_tensor.options().dtype(at::kInt), output_tensor) :
            npu_preparation::apply_tensor(output_tensor, output_size);

    bitwise_xor_out_npu_nocheck(result, self_input, other_input);
    if (output_tensor.dtype() == at::kBool) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
    }
    return result;
}

at::Tensor bitwise_xor(const at::Tensor &self, const at::Scalar &other)
{
    at::Tensor self_input =
        (self.dtype() == at::kBool) ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt) : self;
    at::Tensor result = npu_preparation::apply_tensor(self_input);
    bitwise_xor_out_npu_nocheck(result, self_input, other);
    if (self.dtype() == at::kBool) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
    }
    return result;
}

at::Tensor &bitwise_xor_(at::Tensor &self, const at::Tensor &other)
{
    return acl_op::bitwise_xor_out(self, other, self);
}

at::Tensor &bitwise_xor_(at::Tensor &self, const at::Scalar &other)
{
    return acl_op::bitwise_xor_out(self, other, self);
}
} // namespace acl_op
