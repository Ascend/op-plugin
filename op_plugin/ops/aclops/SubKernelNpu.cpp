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
at::Tensor &sub_scalar_out_npu(at::Tensor &result, const at::Tensor &self, at::Scalar other, at::Scalar alpha)
{
    // other*alpha
    float other_value = op_plugin::utils::get_scalar_float_value(other);
    float alpha_value = op_plugin::utils::get_scalar_float_value(alpha);
    at::Scalar scalarValue(other_value * alpha_value);

    at_npu::native::OpCommand cmd;
    cmd.Name("Sub").Input(self).Input(scalarValue, self.scalar_type()).Output(result).Run();

    return result;
}

at::Tensor &sub_self_scalar_out_npu(at::Tensor &result, at::Scalar self, const at::Tensor &other, at::Scalar alpha)
{
    at::Tensor other_mul_alpha = op_plugin::utils::is_scalar_one(alpha) ? other : acl_op::mul(other, alpha);

    at_npu::native::OpCommand cmd;
    cmd.Name("Sub").Input(self, other_mul_alpha.scalar_type()).Input(other_mul_alpha).Output(result).Run();

    return result;
}

at::Tensor &sub_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    if (npu_preparation::IsCPUScalar(other)) {
        sub_scalar_out_npu(result, self, other.item(), alpha);
    } else if (npu_preparation::IsCPUScalar(self)) {
        sub_self_scalar_out_npu(result, self.item(), other, alpha);
    } else {
        at::Tensor other_mul_result = other;
        if (!op_plugin::utils::is_scalar_one(alpha)) {
            other_mul_result = at::mul(other, alpha);
        }

        at_npu::native::OpCommand cmd;
        cmd.Name("Sub").Expect(unified_result).Input(self).Input(other_mul_result).Output(result).Run();
    }

    return result;
}
} // namespace

at::Tensor &sub_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &result)
{
    at::Tensor output_tensor = npu_preparation::is_scalar_wrapped_to_tensor(self) ? other : self;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = (self.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;
    npu_preparation::CheckOut({self_cp}, result, npu_preparation::get_tensor_npu_format(output_tensor), result_type,
                              output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        sub_out_npu_nocheck(contiguous_result, self_cp, other_cp, alpha);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        sub_out_npu_nocheck(result, self_cp, other_cp, alpha);
    }

    return result;
}

at::Tensor sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor output_tensor = is_self_wrapped ? other : self;

    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = (self.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, output_tensor.options().dtype(result_type), npu_preparation::get_tensor_npu_format(output_tensor));
    sub_out_npu_nocheck(result, self_cp, other_cp, alpha);
    return result;
}

at::Tensor sub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    sub_scalar_out_npu(result, self, other, alpha);
    return result;
}

at::Tensor &sub_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    at::ScalarType result_type = at::native::result_type(self, other);
    at::ScalarType self_type = self.scalar_type();
    TORCH_CHECK(canCast(result_type, self_type), "result type ", result_type,
                " can't be cast to the desired output type ", self_type, OPS_ERROR(ErrCode::TYPE));
    at::Tensor self_cp = (self_type != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;

    npu_preparation::CheckMemory({self_cp, other_cp}, {self_cp});
    if (!npu_utils::check_match(&self_cp)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self_cp);
        at::Tensor result = sub_out_npu_nocheck(contiguous_self, contiguous_self, other_cp, alpha);
        npu_utils::format_fresh_view(self_cp, result);
    } else {
        sub_out_npu_nocheck(self_cp, self_cp, other_cp, alpha);
    }

    if (self_type == result_type) {
        self = self_cp;
    } else {
        self.copy_(self_cp);
    }
    return self;
}

at::Tensor &sub_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        at::Tensor result = sub_scalar_out_npu(contiguous_self, contiguous_self, other, alpha);
        npu_utils::format_fresh_view(self, result);
    } else {
        sub_scalar_out_npu(self, self, other, alpha);
    }

    return self;
}
} // namespace acl_op
