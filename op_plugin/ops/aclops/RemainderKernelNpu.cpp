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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& remainder_out_scalar_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorMod")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();

    return result;
}

at::Tensor& remainder_out_scalar_npu_nocheck(
    at::Tensor& result,
    const at::Scalar& self,
    const at::Tensor& other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorMod")
        .Input(self, other.scalar_type())
        .Input(other)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& remainder_out_tensor_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other)
{
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorMod")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& remainder_out_scalar(
    at::Tensor& result,
    const at::Scalar& self,
    const at::Tensor& other)
{
    at::ScalarType calculate_type = at::native::result_type(other, self);
    at::ScalarType result_type = result.scalar_type();
    TORCH_CHECK(canCast(calculate_type, result_type), "result type ", calculate_type,
                " can't be cast to the desired output type ", result_type, OPS_ERROR(ErrCode::TYPE));

    at::Tensor other_cast = (other.dtype() == calculate_type) ? other :
        at_npu::native::custom_ops::npu_dtype_cast(other, calculate_type);
    at::Tensor result_cast = (result_type == calculate_type) ? result :
        at_npu::native::custom_ops::npu_dtype_cast(result, calculate_type);

    npu_preparation::CheckOut(
        {other},
        result,
        result,
        other.sizes());

    if (!npu_utils::check_match(&result_cast)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
        remainder_out_scalar_npu_nocheck(contiguous_result, self, other_cast);
        npu_utils::format_fresh_view(result_cast, contiguous_result);
    } else {
        remainder_out_scalar_npu_nocheck(result_cast, self, other_cast);
    }

    if (result_type != calculate_type) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
        result.copy_(result_cast);
    }
    return result;
}
} // namespace

at::Tensor& remainder_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        result,
        self.sizes());

    at::ScalarType calculate_type = at::native::result_type(self, other);
    at::ScalarType result_type = result.scalar_type();
    TORCH_CHECK(canCast(calculate_type, result_type), "result type ", calculate_type,
                " can't be cast to the desired output type ", result_type, OPS_ERROR(ErrCode::TYPE));

    at::Tensor self_cast = (self.dtype() == calculate_type) ? self :
        at_npu::native::custom_ops::npu_dtype_cast(self, calculate_type);
    at::Tensor result_cast = (result_type == calculate_type) ? result :
        at_npu::native::custom_ops::npu_dtype_cast(result, calculate_type);
    if (!npu_utils::check_match(&result_cast)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
        remainder_out_scalar_npu_nocheck(contiguous_result, self_cast, other);
        npu_utils::format_fresh_view(result_cast, contiguous_result);
    } else {
        remainder_out_scalar_npu_nocheck(result_cast, self_cast, other);
    }

    if (result_type != calculate_type) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
        result.copy_(result_cast);
    }
    return result;
}

at::Tensor& remainder_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::remainder_out(self, other.item(), result);
    } else if (npu_preparation::IsCPUScalar(self)) {
        return remainder_out_scalar(result, self.item(), other);
    } else {
        auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
        npu_preparation::CheckOut(
            {self, other},
            result,
            result,
            output_size);

        at::ScalarType calculate_type = at::native::result_type(self, other);
        at::ScalarType result_type = result.scalar_type();
        TORCH_CHECK(canCast(calculate_type, result_type), "result type ", calculate_type,
                    " can't be cast to the desired output type ", result_type, OPS_ERROR(ErrCode::TYPE));

        TORCH_CHECK(self.device() == other.device(),
                    "Expected all tensors to be on the same device, but found at least two devices, ",
                    self.device(), " and ", other.device(), OPS_ERROR(ErrCode::PARAM));

        at::Tensor self_cast =
            (self.dtype() == calculate_type) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, calculate_type);
        at::Tensor other_cast =
            (other.dtype() == calculate_type) ? other : at_npu::native::custom_ops::npu_dtype_cast(other, calculate_type);
        at::Tensor result_cast =
            (result_type == calculate_type) ? result : at_npu::native::custom_ops::npu_dtype_cast(result, calculate_type);
        if (!npu_utils::check_match(&result_cast)) {
            at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
            remainder_out_tensor_npu_nocheck(contiguous_result, self_cast, other_cast);
            npu_utils::format_fresh_view(result_cast, contiguous_result);
        } else {
            remainder_out_tensor_npu_nocheck(result_cast, self_cast, other_cast);
        }

        if (result_type != calculate_type) {
            result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
            result.copy_(result_cast);
        }
        return result;
    }
}

at::Tensor remainder(const at::Tensor& self, const at::Tensor& other)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::remainder(self, other.item());
    } else if (npu_preparation::IsCPUScalar(self)) {
        return acl_op::remainder(self.item(), other);
    } else {
        TORCH_CHECK(self.device() == other.device(),
                    "Expected all tensors to be on the same device, but found at least two devices, ",
                    self.device(), " and ", other.device(), OPS_ERROR(ErrCode::PARAM));

        at::ScalarType calculate_type = at::native::result_type(self, other);
        at::Tensor self_cast =
            (self.dtype() == calculate_type) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, calculate_type);
        at::Tensor other_cast =
            (other.dtype() == calculate_type) ? other : at_npu::native::custom_ops::npu_dtype_cast(other, calculate_type);

        auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
        at::Tensor result = npu_preparation::apply_tensor(self_cast, output_size);
        remainder_out_tensor_npu_nocheck(result, self_cast, other_cast);
        return result;
    }
}

at::Tensor remainder(const at::Tensor& self, const at::Scalar& other)
{
    at::ScalarType calculate_type = at::native::result_type(self, other);
    at::Tensor self_cast = (self.dtype() == calculate_type) ? self :
        at_npu::native::custom_ops::npu_dtype_cast(self, calculate_type);
    at::Tensor result = npu_preparation::apply_tensor(self_cast);
    remainder_out_scalar_npu_nocheck(result, self_cast, other);
    return result;
}

at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other)
{
    return acl_op::remainder_out(self, other, self);
}

at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other)
{
    return acl_op::remainder_out(self, other, self);
}

at::Tensor remainder(const at::Scalar& self, const at::Tensor& other)
{
    at::ScalarType calculate_type = at::native::result_type(other, self);
    at::Tensor other_cast = (other.dtype() == calculate_type) ? other :
        at_npu::native::custom_ops::npu_dtype_cast(other, calculate_type);
    at::Tensor result = npu_preparation::apply_tensor(other_cast);
    remainder_out_scalar_npu_nocheck(result, self, other_cast);
    return result;
}
} // namespace acl_op
