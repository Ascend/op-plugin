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
at::Tensor &floor_divide_scalar_out_nocheck(at::Tensor &result, const at::Tensor &self, at::Scalar other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorDiv").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &floor_divide_scalar_out_nocheck(at::Tensor &result, at::Scalar self, const at::Tensor &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FloorDiv").Input(self, other.scalar_type()).Input(other).Output(result).Run();
    return result;
}

at::Tensor &floor_divide_out_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    if (npu_preparation::IsCPUScalar(other)) {
        floor_divide_scalar_out_nocheck(result, self, other.item());
    } else if (npu_preparation::IsCPUScalar(self)) {
        floor_divide_scalar_out_nocheck(result, self.item(), other);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("FloorDiv").Input(self).Input(other).Output(result).Run();
    }
    return result;
}

at::Tensor &check_self_dtype_npu(at::Tensor &self)
{
    if (self.dtype() == at::kBool || self.dtype() == at::kInt) {
        self = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);
    }
    return self;
}
} // namespace

at::Tensor &floor_divide_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self, other}, result, result, output_size);

    at::ScalarType result_type = result.scalar_type();
    at::ScalarType high_type = at::native::result_type(self, other);
    TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
        " can't be cast to the desired output type ", result_type,
        OPS_ERROR(ErrCode::TYPE));

    at::ScalarType cal_type = op_plugin::utils::get_divide_calculate_type(self, other);
    at::Tensor self_cast = self.scalar_type() != cal_type ? self.to(cal_type) : self;
    at::Tensor other_cast = other.scalar_type() != cal_type ? other.to(cal_type) : other;

    bool result_type_is_cal_type = result_type == cal_type;
    at::Tensor result_cast =
        result_type_is_cal_type ? result : at_npu::native::custom_ops::npu_dtype_cast(result, cal_type);
    if (!npu_utils::check_match(&result_cast)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cast);
        floor_divide_out_nocheck(contiguous_result, self_cast, other_cast);
        npu_utils::format_fresh_view(result_cast, contiguous_result);
    } else {
        floor_divide_out_nocheck(result_cast, self_cast, other_cast);
    }

    if (!result_type_is_cal_type) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result_type);
        result.copy_(result_cast);
    }
    return result;
}

at::Tensor floor_divide(const at::Tensor &self, const at::Tensor &other)
{
    at::ScalarType cal_type = op_plugin::utils::get_divide_calculate_type(self, other);
    at::Tensor self_cast = self.scalar_type() != cal_type ? self.to(cal_type) : self;
    at::Tensor other_cast = other.scalar_type() != cal_type ? other.to(cal_type) : other;

    bool is_self_wrapped =
        npu_preparation::is_scalar_wrapped_to_tensor(self_cast) || npu_preparation::IsCPUScalar(self_cast);
    at::Tensor output_tensor = is_self_wrapped ? other_cast : self_cast;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self_cast, other_cast);
    at::Tensor result = npu_preparation::apply_tensor(output_tensor, output_size);
    floor_divide_out_nocheck(result, self_cast, other_cast);

    at::ScalarType high_type = at::native::result_type(self, other);
    if (cal_type != high_type) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, high_type);
    }
    return result;
}

at::Tensor floor_divide(const at::Tensor &self, const at::Scalar &other)
{
    at::Tensor self_cast = self;
    check_self_dtype_npu(self_cast);
    at::Tensor format_cast_self = npu_preparation::CastBackToOriFormat(self_cast);
    auto output_size = format_cast_self.sizes();
    at::Tensor result = npu_preparation::apply_tensor(self_cast, output_size);
    floor_divide_scalar_out_nocheck(result, format_cast_self, other);
    return result;
}

at::Tensor &floor_divide_(at::Tensor &self, const at::Tensor &other)
{
    return acl_op::floor_divide_out(self, other, self);
}

at::Tensor &floor_divide_(at::Tensor &self, const at::Scalar &other)
{
    check_self_dtype_npu(self);
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        floor_divide_scalar_out_nocheck(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        floor_divide_scalar_out_nocheck(self, self, other);
    }
    return self;
}
} // namespace acl_op
