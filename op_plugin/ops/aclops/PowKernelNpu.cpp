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
using npu_utils = at_npu::native::NpuUtils;

namespace {
// pow.Tensor_Tensor_out
at::Tensor& pow_tensor_tensor_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& exp)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Pow")
        .Input(self)
        .Input(exp)
        .Output(result)
        .Run();

    return result;
}

// pow.Tensor_Scalar_out
at::Tensor& pow_tensor_scalar_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar exp)
{
    at_npu::native::OpCommand cmd;
    if (exp.toFloat() == 2.0) {
        cmd.Name("Square")
            .Input(self)
            .Output(result)
            .Run();
    } else {
        cmd.Name("Pow")
            .Input(self)
            .Input(exp, self.scalar_type())
            .Output(result)
            .Run();
    }
    return result;
}

// pow.Scalar_out
at::Tensor& pow_scalar_out_npu_nocheck(at::Tensor& result, at::Scalar self, const at::Tensor& exp)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Pow")
        .Input(self, exp.scalar_type())
        .Input(exp)
        .Output(result)
        .Run();

    return result;
}
} // namespace

// pow.Tensor_Tensor_out
at::Tensor& pow_out(const at::Tensor& self, const at::Tensor& exp, at::Tensor& result)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, exp);
    npu_preparation::CheckOut(
        {self, exp},
        result,
        self,
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        pow_tensor_tensor_out_npu_nocheck(contiguous_result, self, exp);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        pow_tensor_tensor_out_npu_nocheck(result, self, exp);
    }
    return result;
}

// pow.Tensor_Scalar_out
at::Tensor& pow_out(const at::Tensor& self, const at::Scalar& exp, at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        self);

    auto exp_value = exp.toFloat();
    if (exp_value == 0.0) {
        return result.fill_(1);
    } else if (exp_value == 1.0) {
        return result.copy_(self);
    }

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        pow_tensor_scalar_out_npu_nocheck(contiguous_result, self, exp);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        pow_tensor_scalar_out_npu_nocheck(result, self, exp);
    }
    return result;
}

// pow.Scalar_out
at::Tensor& pow_out(const at::Scalar& self, const at::Tensor& exp, at::Tensor& result)
{
    npu_preparation::CheckOut(
        {exp},
        result,
        exp);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        pow_scalar_out_npu_nocheck(contiguous_result, self, exp);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        pow_scalar_out_npu_nocheck(result, self, exp);
    }
    return result;
}

at::Tensor pow(const at::Tensor& self, const at::Tensor& exp)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, exp);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    pow_tensor_tensor_out_npu_nocheck(result, self, exp);
    return result;
}

at::Tensor pow(const at::Tensor& self, const at::Scalar& exp)
{
    auto result_type = at::result_type(self, exp);
    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(result_type));

    auto exp_value = exp.toFloat();
    if (exp_value == 0.0) {
        return result.fill_(1);
    } else if (exp_value == 1.0) {
        return result.copy_(self);
    }

    at::Tensor self_copy = (self.scalar_type() != result_type) ? at_npu::native::custom_ops::npu_dtype_cast(self, result_type) : self;
    pow_tensor_scalar_out_npu_nocheck(result, self_copy, exp);
    return result;
}

at::Tensor pow(const at::Scalar& self, const at::Tensor& exp)
{
    auto result_type = at::result_type(exp, self);
    at::Tensor result = npu_preparation::apply_tensor(exp, exp.options().dtype(result_type));
    at::Tensor exp_copy = (exp.scalar_type() != result_type) ? at_npu::native::custom_ops::npu_dtype_cast(exp, result_type) : exp;
    pow_scalar_out_npu_nocheck(result, self, exp_copy);
    return result;
}

at::Tensor& pow_(at::Tensor& self, const at::Tensor& exp)
{
    acl_op::pow_out(self, exp, self);
    return self;
}

at::Tensor& pow_(at::Tensor& self, const at::Scalar& exp)
{
    acl_op::pow_out(self, exp, self);
    return self;
}
} // namespace at_npu
