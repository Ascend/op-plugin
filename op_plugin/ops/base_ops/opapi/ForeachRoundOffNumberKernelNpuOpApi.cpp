// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const char ROUND_MODE_FLOOR = char(2);
const char ROUND_MODE_CEIL = char(3);
const char ROUND_MODE_ROUND = char(4);
const char ROUND_MODE_TRUNC = char(5);
const char ROUND_MODE_FRAC = char(7);

bool is_integral_tensor_list(at::TensorList self)
{
    auto scalarType = self[0].scalar_type();
    return (scalarType == at::ScalarType::Byte
    || scalarType == at::ScalarType::Char
    || scalarType == at::ScalarType::Short
    || scalarType == at::ScalarType::Int
    || scalarType == at::ScalarType::Long);
}

void exec_npu_cmd_(at::TensorList self, const char roundMode)
{
    if (is_integral_tensor_list(self)) {
        return;
    }
    at::Tensor round_mode_scalar_tensor = at_npu::native::OpPreparation::copy_scalar_to_device(
        roundMode, at::ScalarType::Char, self[0].device());
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber, self, round_mode_scalar_tensor, self);
}

std::vector<at::Tensor> exec_npu_cmd(at::TensorList self, const char roundMode)
{
    bool is_integral = is_integral_tensor_list(self);
    auto scalarType = self[0].scalar_type();
    // construct the output tensorlist of the NPU
    std::vector<at::Tensor> result;
    for (uint32_t i = 0; i < self.size(); i++) {
        at::Tensor tensor = self[i];
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            at_npu::native::OpPreparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalarType))
        );
        if (is_integral) {
            result[i] = tensor.clone();
        }
    }

    if (is_integral) {
        return result;
    }

    at::TensorList result_ = at::TensorList(result);

    at::Tensor round_mode_scalar_tensor = at_npu::native::OpPreparation::copy_scalar_to_device(
        roundMode, at::ScalarType::Char, self[0].device());
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber, self, round_mode_scalar_tensor, result_);
    return result;
}

bool if_use_slow_route(at::TensorList tensors, const bool isFrac)
{
    at::native::check_foreach_api_restrictions(tensors);
    return !at_npu::native::env::CheckJitDisable() ||
           !at::native::can_use_fast_route(tensors) ||
           (isFrac && at::native::has_integral_tensor(tensors, true));
}

bool if_use_slow_route(at::TensorList tensors)
{
    return if_use_slow_route(tensors, false);
}

void _foreach_floor_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_floor_slow_(self);
    }
    exec_npu_cmd_(self, ROUND_MODE_FLOOR);
}

std::vector<at::Tensor> _foreach_floor(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_floor_slow(self);
    }
    return exec_npu_cmd(self, ROUND_MODE_FLOOR);
}

void _foreach_ceil_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_ceil_slow_(self);
    }
    exec_npu_cmd_(self, ROUND_MODE_CEIL);
}

std::vector<at::Tensor> _foreach_ceil(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_ceil_slow(self);
    }
    return exec_npu_cmd(self, ROUND_MODE_CEIL);
}

void _foreach_round_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_round_slow_(self);
    }
    exec_npu_cmd_(self, ROUND_MODE_ROUND);
}

std::vector<at::Tensor> _foreach_round(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_round_slow(self);
    }
    return exec_npu_cmd(self, ROUND_MODE_ROUND);
}

void _foreach_trunc_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_trunc_slow_(self);
    }
    exec_npu_cmd_(self, ROUND_MODE_TRUNC);
}

std::vector<at::Tensor> _foreach_trunc(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_trunc_slow(self);
    }
    return exec_npu_cmd(self, ROUND_MODE_TRUNC);
}

void _foreach_frac_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_frac_slow_(self);
    }
    exec_npu_cmd_(self, ROUND_MODE_FRAC);
}

std::vector<at::Tensor> _foreach_frac(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_frac_slow(self);
    }
    return exec_npu_cmd(self, ROUND_MODE_FRAC);
}
} // namespace op_api
