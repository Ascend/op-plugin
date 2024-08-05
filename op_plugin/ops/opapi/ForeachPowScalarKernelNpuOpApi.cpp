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
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void checkFloat(at::ScalarType scalar_type)
{
    TORCH_CHECK(scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float ||
                scalar_type == at::ScalarType::Int || scalar_type == at::ScalarType::BFloat16,
                "input must be half, float, int32 or bfloat16");
}

void _split_and_exec_npu_cmd_pow_kernel(const at::TensorList self, const at::Scalar& scalar, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = self.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, self);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachPowScalar, self, scalar_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_self(self.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachPowScalar, temp_self, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_self(self.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachPowScalar, temp_self, scalar_, temp_result);
    }
}

std::vector<at::Tensor> _foreach_pow(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_pow_scalar_kernel_slow(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    checkFloat(scalar_type);

    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_pow_kernel(self, scalar, result_, false);
    return result;
}

void _foreach_pow_(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_pow_scalar_kernel_slow_(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    checkFloat(scalar_type);

    _split_and_exec_npu_cmd_pow_kernel(self, scalar, self, true);
}
#endif
}  // namespace op_api
