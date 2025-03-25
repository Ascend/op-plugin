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

#include <climits>
#include <cfloat>
#include <ATen/native/TypeProperties.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& clamp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar min,
    at::Scalar max)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ClipByValueV2")
        .Input(self)
        .Input(min, self.scalar_type())
        .Input(max, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}

at::Tensor& clamp_min_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar min)
{
    // Set max according to self.dtype()
    at::Scalar max;
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        max = INT_MAX;
    } else if (self.dtype() == at::kFloat) {
        max = FLT_MAX;
    } else {
        max = NPU_HALF_MAX;
    }
    clamp_out_npu_nocheck(result, self, min, max);
    return result;
}

at::Tensor& clamp_max_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar max)
{
    // Set max according to self.dtype()
    at::Scalar min;
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        min = INT_MIN;
    } else if (self.dtype() == at::kFloat) {
        min = -FLT_MAX;
    } else {
        min = NPU_HALF_MIN;
    }
    clamp_out_npu_nocheck(result, self, min, max);
    return result;
}

at::Tensor& clamp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& min,
    const at::Tensor& max)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ClipByValueV2")
        .Input(self)
        .Input(min)
        .Input(max)
        .Output(result)
        .Run();
    return result;
}

// clamp.Tensor
at::Tensor& clamp_min_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& min)
{
    at::Tensor max;
    at::Tensor ones_tensor = at::ones(self.sizes(), self.options());
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        max = ones_tensor * INT_MAX;
    } else if (self.dtype() == at::kFloat) {
        max = ones_tensor * FLT_MAX;
    } else {
        max = ones_tensor * NPU_HALF_MAX;
    }
    return clamp_out_npu_nocheck(result, self, min, max);
}

at::Tensor& clamp_max_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& max)
{
    // Set min according to self.dtype()
    at::Tensor min;
    at::Tensor ones_tensor = at::ones(self.sizes(), self.options());
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        min = ones_tensor * INT_MIN;
    } else if (self.dtype() == at::kFloat) {
        min = ones_tensor * (-FLT_MAX);
    } else {
        min = ones_tensor * NPU_HALF_MIN;
    }
    return clamp_out_npu_nocheck(result, self, min, max);
}
} // namespace

at::Tensor& clamp_min_out(
    const at::Tensor& self,
    const at::Scalar& min,
    at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        self);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        clamp_min_out_npu_nocheck(contiguous_result, self, min);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        clamp_min_out_npu_nocheck(result, self, min);
    }
    return result;
}

at::Tensor& clamp_max_out(
    const at::Tensor& self,
    const at::Scalar& max,
    at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        self);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        clamp_max_out_npu_nocheck(contiguous_result, self, max);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        clamp_max_out_npu_nocheck(result, self, max);
    }
    return result;
}

at::Tensor& clamp_out(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max,
    at::Tensor& result)
{
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None"
                + OPS_ERROR(ErrCode::VALUE));
    if (!min.has_value()) {
        at::Scalar max_value = max.value();
        return acl_op::clamp_max_out(self, max_value, result);
    } else if (!max.has_value()) {
        at::Scalar min_value = min.value();
        return acl_op::clamp_min_out(self, min_value, result);
    } else {
        at::Scalar min_value = min.value();
        at::Scalar max_value = max.value();
        npu_preparation::CheckOut(
            {self},
            result,
            self);
        if (!npu_utils::check_match(&result)) {
            at::Tensor contiguous_result = npu_utils::format_contiguous(result);
            clamp_out_npu_nocheck(contiguous_result, self, min_value, max_value);
            npu_utils::format_fresh_view(result, contiguous_result);
        } else {
            clamp_out_npu_nocheck(result, self, min_value, max_value);
        }
        return result;
    }
}

at::Tensor clamp_min(const at::Tensor& self, const at::Scalar& min)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_min_out(self, min, result);
}

at::Tensor& clamp_min_(at::Tensor& self, const at::Scalar& min)
{
    return acl_op::clamp_min_out(self, min, self);
}

at::Tensor clamp_max(const at::Tensor& self, const at::Scalar& max)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_max_out(self, max, result);
}

at::Tensor& clamp_max_(at::Tensor& self, const at::Scalar& max)
{
    return acl_op::clamp_max_out(self, max, self);
}

at::Tensor clamp(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_out(self, min, max, result);
}

at::Tensor& clamp_(
    at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max)
{
    return acl_op::clamp_out(self, min, max, self);
}

at::Tensor& clamp_min_out(
    const at::Tensor& self,
    const at::Tensor& min,
    at::Tensor& result)
{
    auto high_dtype = at::native::result_type(self, min);
    auto result_dtype = result.scalar_type();
    TORCH_CHECK(canCast(high_dtype, result_dtype),
                "result type ", high_dtype, " can't be cast to the desired output type ", result_dtype, OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(result_dtype != at::kBool, "'clamp_npu' not implemented for 'Bool'" + OPS_ERROR(ErrCode::TYPE));

    at::Tensor self_cp = self.scalar_type() == result_dtype ? self : at_npu::native::custom_ops::npu_dtype_cast(self, result_dtype);
    at::Tensor min_cp = min.scalar_type() == result_dtype ? min : at_npu::native::custom_ops::npu_dtype_cast(min, result_dtype);
    if (min_cp.sizes() != self.sizes()) {
        min_cp = acl_op::npu_broadcast(min_cp, self.sizes());
    }
    npu_preparation::CheckOut(
        {self_cp, min_cp},
        result,
        self_cp);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        clamp_min_out_npu_nocheck(contiguous_result, self_cp, min_cp);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        clamp_min_out_npu_nocheck(result, self_cp, min_cp);
    }
    return result;
}

at::Tensor& clamp_max_out(
    const at::Tensor& self,
    const at::Tensor& max,
    at::Tensor& result)
{
    auto high_dtype = at::native::result_type(self, max);
    auto result_dtype = result.scalar_type();
    TORCH_CHECK(canCast(high_dtype, result_dtype),
                "result type ", high_dtype, " can't be cast to the desired output type ", result_dtype,
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(result_dtype != at::kBool, "'clamp_npu' not implemented for 'Bool'"
                + OPS_ERROR(ErrCode::TYPE));

    at::Tensor self_cp = self.scalar_type() == result_dtype ? self : at_npu::native::custom_ops::npu_dtype_cast(self, result_dtype);
    at::Tensor max_cp = max.scalar_type() == result_dtype ? max : at_npu::native::custom_ops::npu_dtype_cast(max, result_dtype);
    if (max_cp.sizes() != self.sizes()) {
        max_cp = acl_op::npu_broadcast(max_cp, self.sizes());
    }
    npu_preparation::CheckOut(
        {self_cp, max_cp},
        result,
        self_cp);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        clamp_max_out_npu_nocheck(contiguous_result, self_cp, max_cp);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        clamp_max_out_npu_nocheck(result, self_cp, max_cp);
    }
    return result;
}

at::Tensor& clamp_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& min,
    const c10::optional<at::Tensor>& max,
    at::Tensor& result)
{
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp: At least one of 'min' or 'max' must not be None"
                + OPS_ERROR(ErrCode::VALUE));
    if (!min.has_value()) {
        const at::Tensor& max_value = max.value();
        return acl_op::clamp_max_out(self, max_value, result);
    } else if (!max.has_value()) {
        const at::Tensor& min_value = min.value();
        return acl_op::clamp_min_out(self, min_value, result);
    } else {
        const at::Tensor& min_value = min.value();
        const at::Tensor& max_value = max.value();
        at::TensorList tensors = {self, min_value, max_value};
        auto high_dtype = at::native::result_type(tensors);
        auto result_dtype = result.scalar_type();
        TORCH_CHECK(canCast(high_dtype, result_dtype),
                    "result type ", high_dtype, " can't be cast to the desired output type ", result_dtype,
                    OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(result_dtype != at::kBool, "'clamp_npu' not implemented for 'Bool'"
                    + OPS_ERROR(ErrCode::TYPE));

        at::Tensor self_cp = self.scalar_type() == result_dtype ? self :
            at_npu::native::custom_ops::npu_dtype_cast(self, result_dtype);
        at::Tensor min_value_cp = min_value.scalar_type() == result_dtype ? min_value :
            at_npu::native::custom_ops::npu_dtype_cast(min_value, result_dtype);
        at::Tensor max_value_cp = max_value.scalar_type() == result_dtype ? max_value :
            at_npu::native::custom_ops::npu_dtype_cast(max_value, result_dtype);
        if (max_value_cp.sizes() != self.sizes()) {
            max_value_cp = acl_op::npu_broadcast(max_value_cp, self.sizes());
        }
        if (min_value_cp.sizes() != self.sizes()) {
            min_value_cp = acl_op::npu_broadcast(min_value_cp, self.sizes());
        }
        npu_preparation::CheckOut(
            {self_cp, min_value_cp, max_value_cp},
            result,
            self_cp);
        if (!npu_utils::check_match(&result)) {
            at::Tensor contiguous_result = npu_utils::format_contiguous(result);
            clamp_out_npu_nocheck(contiguous_result, self_cp, min_value_cp, max_value_cp);
            npu_utils::format_fresh_view(result, contiguous_result);
        } else {
            clamp_out_npu_nocheck(result, self_cp, min_value_cp, max_value_cp);
        }
        return result;
    }
}

at::Tensor clamp_min(const at::Tensor& self, const at::Tensor& min)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_min_out(self, min, result);
}

at::Tensor& clamp_min_(at::Tensor& self, const at::Tensor& min)
{
    return acl_op::clamp_min_out(self, min, self);
}

at::Tensor clamp_max(const at::Tensor& self, const at::Tensor& max)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_max_out(self, max, result);
}

at::Tensor& clamp_max_(at::Tensor& self, const at::Tensor& max)
{
    return acl_op::clamp_max_out(self, max, self);
}

at::Tensor clamp(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& min,
    const c10::optional<at::Tensor>& max)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return acl_op::clamp_out(self, min, max, result);
}

at::Tensor& clamp_(
    at::Tensor& self,
    const c10::optional<at::Tensor>& min,
    const c10::optional<at::Tensor>& max)
{
    return acl_op::clamp_out(self, min, max, self);
}
} // namespace acl_op
