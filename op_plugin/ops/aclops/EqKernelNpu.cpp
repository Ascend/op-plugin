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
at::Tensor& eq_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other)
{
    auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("Equal")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& eq_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Equal")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}

at::ScalarType get_eq_calculate_type(const at::Tensor& self, const at::Tensor& other)
{
    at::ScalarType calculate_type = at::native::result_type(self, other);
    if (calculate_type == at::kInt) {
        calculate_type = at::kFloat;
    }
    return calculate_type;
}

at::ScalarType get_eq_calculate_type(const at::Tensor& self, const at::Scalar& other)
{
    at::ScalarType calculate_type = at::native::result_type(self, other);
    if (calculate_type == at::kInt) {
        calculate_type = at::kFloat;
    }
    return calculate_type;
}
} // namespace

at::Tensor& eq_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::eq_out(self, other.item(), result);
    } else if (npu_preparation::IsCPUScalar(self)) {
        return acl_op::eq_out(other, self.item(), result);
    } else {
        auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
        npu_preparation::CheckOut(
            {self, other},
            result,
            result,
            output_size);

        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device(),
            OPS_ERROR(ErrCode::PARAM));

        auto calculate_type = get_eq_calculate_type(self, other);
        auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
        auto other_cast = op_plugin::utils::get_cast_input(other, calculate_type);

        auto result_type = result.scalar_type();
        at::Tensor result_cast = (result_type != at::kBool) ?
            at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool) : result;
        if (!npu_utils::check_match(&result_cast)) {
            at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
            eq_out_npu_nocheck(contiguous_result, self_cast, other_cast);
            npu_utils::format_fresh_view(result_cast, contiguous_result);
        } else {
            eq_out_npu_nocheck(result_cast, self_cast, other_cast);
        }

        if (result_type != at::kBool) {
            result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
            result.copy_(result_cast);
        }
        return result;
    }
}

at::Tensor& eq_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& result)
{
    auto calculate_type = get_eq_calculate_type(self, other);
    auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
    npu_preparation::CheckOut(
        {self_cast},
        result,
        result,
        self.sizes());

    auto result_type = result.scalar_type();
    at::Tensor result_cast = (result_type != at::kBool) ?
        at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool) : result;
    if (!npu_utils::check_match(&result_cast)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
        eq_out_npu_nocheck(contiguous_result, self_cast, other);
        npu_utils::format_fresh_view(result_cast, contiguous_result);
    } else {
        eq_out_npu_nocheck(result_cast, self_cast, other);
    }

    if (result_type != at::kBool) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
        result.copy_(result_cast);
    }
    return result;
}

at::Tensor eq(
    const at::Tensor& self,
    const at::Tensor& other)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::eq(self, other.item());
    } else if (npu_preparation::IsCPUScalar(self)) {
        return acl_op::eq(other, self.item());
    } else {
        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device(),
            OPS_ERROR(ErrCode::PARAM));

        auto calculate_type = get_eq_calculate_type(self, other);
        auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
        auto other_cast = op_plugin::utils::get_cast_input(other, calculate_type);

        auto output_size = op_infer::broadcast_ops_npu_output_size(self_cast, other_cast);
        at::Tensor result = npu_preparation::apply_tensor_with_sizes(
            output_size,
            self_cast.options().dtype(at::kBool));

        eq_out_npu_nocheck(result, self_cast, other_cast);
        return result;
    }
}

at::Tensor eq(
    const at::Tensor& self,
    const at::Scalar& other)
{
    auto calculate_type = get_eq_calculate_type(self, other);
    auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(at::kBool));
    eq_out_npu_nocheck(result, self_cast, other);
    return result;
}

at::Tensor& eq_(
    at::Tensor& self,
    const at::Tensor& other)
{
    return acl_op::eq_out(self, other, self);
}

at::Tensor& eq_(
    at::Tensor& self,
    const at::Scalar& other)
{
    return acl_op::eq_out(self, other, self);
}
} // namespace acl_op
