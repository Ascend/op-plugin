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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_pow(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_pow_scalar_kernel_slow(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32", OPS_ERROR(ErrCode::TYPE));
    }
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, self[0].scalar_type(), self[0].device());
    EXEC_NPU_CMD(aclnnForeachPowScalar, self, scalar_tensor, result_);
    return result;
}

void _foreach_pow_(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_pow_scalar_kernel_slow_(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32", OPS_ERROR(ErrCode::TYPE));
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, self[0].scalar_type(), self[0].device());
    EXEC_NPU_CMD(aclnnForeachPowScalar, self, scalar_tensor, self);
}
} // namespace op_api
