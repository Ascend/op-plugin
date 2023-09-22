// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

void exec_npu_cmd(at::TensorList self, const char roundMode)
{
    at::ScalarType scalarType = self[0].scalar_type();
    if (scalarType == at::ScalarType::Byte
    || scalarType == at::ScalarType::Char
    || scalarType == at::ScalarType::Short
    || scalarType == at::ScalarType::Int
    || scalarType == at::ScalarType::Long) {
        return;
    }

    at::Tensor round_mode_scalar_tensor = at_npu::native::OpPreparation::copy_scalar_to_device(
        roundMode, at::ScalarType::Char);
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnForeachRoundOffNumberInplace, self, round_mode_scalar_tensor);
}

bool if_use_slow_route(at::TensorList tensors)
{
    at::native::check_foreach_api_restrictions(tensors);
    return !at::native::can_use_fast_route(tensors) || at::native::has_integral_tensor(tensors, true);
}

void _foreach_floor_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_floor_slow_(self);
    }
    exec_npu_cmd(self, ROUND_MODE_FLOOR);
}

void _foreach_ceil_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_ceil_slow_(self);
    }
    exec_npu_cmd(self, ROUND_MODE_CEIL);
}

void _foreach_round_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_round_slow_(self);
    }
    exec_npu_cmd(self, ROUND_MODE_ROUND);
}

void _foreach_trunc_(at::TensorList self)
{
    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_trunc_slow_(self);
    }
    exec_npu_cmd(self, ROUND_MODE_TRUNC);
}
} // namespace op_api