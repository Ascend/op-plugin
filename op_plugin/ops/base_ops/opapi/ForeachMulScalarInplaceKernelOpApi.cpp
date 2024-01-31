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
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32");
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type);
    EXEC_NPU_CMD(aclnnForeachMulScalar, self, scalar_tensor, self);
}
}