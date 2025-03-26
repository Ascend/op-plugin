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
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_addcdiv_v1(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({input, tensors1, tensors2}, scalar) ||
        at::native::has_integral_tensor(input, true)) {
        return at::native::foreach_tensor_addcdiv_scalar_slow(input, tensors1, tensors2, scalar);
    }
    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float", OPS_ERROR(ErrCode::TYPE));
    }
    std::vector<at::Tensor> result;
    result.reserve(input.size());
    for (const at::Tensor &tensor : input) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, input[0].scalar_type(),
                                                                      input[0].device());
    EXEC_NPU_CMD(aclnnForeachAddcdivScalar, input, tensors1, tensors2, scalar_tensor, result_);

    return result;
}

void _foreach_addcdiv_v1_(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({input, tensors1, tensors2}, scalar) ||
        at::native::has_integral_tensor(input, true)) {
        return at::native::foreach_tensor_addcdiv_scalar_slow_(input, tensors1, tensors2, scalar);
    }
    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float", OPS_ERROR(ErrCode::TYPE));
    }

    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, input[0].scalar_type(),
                                                                      input[0].device());
    EXEC_NPU_CMD(aclnnForeachAddcdivScalar, input, tensors1, tensors2, scalar_tensor, input);
}

void _split_and_exec_npu_cmd_addcdiv_scalar(const at::TensorList input,
                                            const at::TensorList tensors1,
                                            const at::TensorList tensors2,
                                            const at::Scalar &scalars,
                                            at::TensorList result,
                                            bool is_inplace)
{
    size_t tensor_count = input.size();
    size_t max_tensor_count = is_inplace ? 16 : 12;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalars, input);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachAddcdivScalarV2, input, tensors1, tensors2, scalar_, result);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_input(input.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachAddcdivScalarV2, temp_input, temp_tensors1, temp_tensors2, scalar_, temp_result);
    }
    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count > 0) {
        at::TensorList temp_input(input.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachAddcdivScalarV2, temp_input, temp_tensors1, temp_tensors2, scalar_, temp_result);
    }
}

std::vector<at::Tensor> _foreach_addcdiv(at::TensorList input,
                                         at::TensorList tensor1,
                                         at::TensorList tensor2,
                                         const at::Scalar &value)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_addcdiv_scalar_slow(input, tensor1, tensor2, value);
    }
    DO_COMPATIBILITY(aclnnForeachAddcdivScalarV2, _foreach_addcdiv_v1(input, tensor1, tensor2, value));
    at::native::check_foreach_api_restrictions(input, tensor1, tensor2);
    if (!at::native::can_use_fast_route({input, tensor1, tensor2}, value) ||
        at::native::has_integral_tensor(input, true)) {
        return at::native::foreach_tensor_addcdiv_scalar_slow(input, tensor1, tensor2, value);
    }
    auto scalar_type = input[0].scalar_type();

    std::vector<at::Tensor> result;
    result.reserve(input.size());
    for (const at::Tensor &tensor : input) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_addcdiv_scalar(input, tensor1, tensor2, value, result_, false);

    return result;
}

void _foreach_addcdiv_(at::TensorList self,
                       at::TensorList tensor1,
                       at::TensorList tensor2,
                       const at::Scalar &value)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_addcdiv_scalar_slow_(self, tensor1, tensor2, value);
    }
    DO_COMPATIBILITY(aclnnForeachAddcdivScalarV2, _foreach_addcdiv_v1_(self, tensor1, tensor2, value));
    at::native::check_foreach_api_restrictions(self, tensor1, tensor2);
    if (!at::native::can_use_fast_route({self, tensor1, tensor2}, value) ||
        at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_addcdiv_scalar_slow_(self, tensor1, tensor2, value);
    }

    _split_and_exec_npu_cmd_addcdiv_scalar(self, tensor1, tensor2, value, self, true);
}
}
