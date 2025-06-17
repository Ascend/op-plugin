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

#include <ATen/native/TypeProperties.h>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "op_plugin/utils/KernelNpuOutputDtype.h"

namespace op_infer {

at::ScalarType angle_out_dtype(const at::Tensor& self)
{
    auto out_dtype = self.scalar_type();
    if (self.is_complex()) {
        out_dtype = self.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    } else if (at::isIntegralType(out_dtype, true)) {
        out_dtype = at::kFloat;
    }
    return out_dtype;
}

at::ScalarType polar_out_dtype(const at::Tensor& abs, const at::Tensor& angle)
{
    at::ScalarType high_type = at::native::result_type(abs, angle);
    if (high_type == at::ScalarType::Float) {
        high_type = at::ScalarType::ComplexFloat;
    } else if (high_type == at::ScalarType::Double) {
        high_type = at::ScalarType::ComplexDouble;
    } else if (high_type == at::ScalarType::Half) {
        high_type = at::ScalarType::ComplexHalf;
    }
    return high_type;
}

at::ScalarType npu_group_quant_dst_type(c10::optional<at::ScalarType> dst_dtype)
{
    at::ScalarType dst_type = c10::value_or_else(dst_dtype, [] {return at::ScalarType::Char;});
    if (dst_type == at::kQInt8) {
        dst_type = at::kChar;
    }
    TORCH_CHECK(dst_type == at::ScalarType::Char || dst_type == at::ScalarType::QUInt4x2,
                "dst_dtype must be Int8 or Int4" + OPS_ERROR(ErrCode::TYPE));
    if (dst_type == at::ScalarType::QUInt4x2) {
        dst_type = at::ScalarType::Int;
    }
    return dst_type;
}

at::ScalarType clamp_out_dtype(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max)
{
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp:At least one of 'min' or 'max' must be not None!");

    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (!min.has_value()) {
        state = at::native::update_result_type_state(max.value(), state);
    } else if (!max.has_value()) {
        state = at::native::update_result_type_state(min.value(), state);
    } else {
        state = at::native::update_result_type_state(max.value(), state);
        state = at::native::update_result_type_state(min.value(), state);
    }

    return at::native::result_type(state);
}

at::ScalarType clamp_scalar_out_dtype(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max)
{
    TORCH_CHECK(min.has_value() || max.has_value(), "torch.clamp:At least one of 'min' or 'max' must be not None!");

    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (!min.has_value()) {
        state = at::native::update_result_type_state(max.value(), state);
    } else if (!max.has_value()) {
        state = at::native::update_result_type_state(min.value(), state);
    } else {
        state = at::native::update_result_type_state(max.value(), state);
        state = at::native::update_result_type_state(min.value(), state);
    }

    return at::native::result_type(state);
}

} // namespace op_infer
