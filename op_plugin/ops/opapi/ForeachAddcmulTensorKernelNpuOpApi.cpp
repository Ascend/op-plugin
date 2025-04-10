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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void _split_and_exec_npu_cmd_addcmul_tensor(const at::TensorList input,
                                            const at::TensorList tensors1,
                                            const at::TensorList tensors2,
                                            at::Tensor scalars,
                                            at::TensorList result,
                                            bool is_inplace)
{
    size_t tensor_count = input.size();
    size_t max_tensor_count = is_inplace ? 16 : 12;
    size_t loop_time = tensor_count / max_tensor_count;
    size_t remaining_count = tensor_count % max_tensor_count;
    size_t data_count = max_tensor_count;
    if (remaining_count > 0) {
        loop_time++;
    }

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachAddcmulScalarList, input, tensors1, tensors2, scalars, result);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        if (i == loop_time - 1 && remaining_count > 0) {
            data_count = remaining_count;
        }
        at::TensorList temp_input(input.data() + i * max_tensor_count, data_count);
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, data_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, data_count);
        at::Tensor temp_scalars = scalars.slice(0, i * max_tensor_count, data_count + i * max_tensor_count);
        at::TensorList temp_result(result.data() + i * max_tensor_count, data_count);
        EXEC_NPU_CMD(aclnnForeachAddcmulScalarList, temp_input, temp_tensors1,
                     temp_tensors2, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_addcmul(const at::TensorList input,
                                         const at::TensorList tensors1,
                                         const at::TensorList tensors2,
                                         const at::Tensor &scalars)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    
    auto scalars_ = at::native::convert_tensor_to_scalar_list(scalars, input.size());
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_addcmul_scalarlist_slow(input, tensors1, tensors2, scalars_);
    }

    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars_);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({input, tensors1, tensors2}) ||
        at::native::has_integral_tensor(input, true)) {
            return at::native::foreach_tensor_addcmul_scalarlist_slow(input, tensors1, tensors2, scalars_);
    }

    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float, int32 or bfloat16" + OPS_ERROR(ErrCode::TYPE));
    }
    std::vector<at::Tensor> result;
    result.reserve(input.size());
    for (const at::Tensor &tensor : input) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    auto scalar_tensor = npu_preparation::copy_tensor_host_to_device(scalars);
    _split_and_exec_npu_cmd_addcmul_tensor(input, tensors1, tensors2, scalar_tensor, result_, false);
    return result;
}

void _foreach_addcmul_(const at::TensorList input,
                       const at::TensorList tensors1,
                       const at::TensorList tensors2,
                       const at::Tensor &scalars)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    
    auto scalars_ = at::native::convert_tensor_to_scalar_list(scalars, input.size());
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_addcmul_scalarlist_slow_(input, tensors1, tensors2, scalars_);
    }

    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars_);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({input, tensors1, tensors2}) ||
        at::native::has_integral_tensor(input, true)) {
            return at::native::foreach_tensor_addcmul_scalarlist_slow_(input, tensors1, tensors2, scalars_);
    }

    at::native::check_foreach_api_restrictions(input, tensors1, tensors2);
    auto scalar_type = input[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float, int32, or bfloat16" + OPS_ERROR(ErrCode::TYPE));
    }
    auto scalar_tensor = npu_preparation::copy_tensor_host_to_device(scalars);

    _split_and_exec_npu_cmd_addcmul_tensor(input, tensors1, tensors2, scalar_tensor, input, true);
}
#endif
}  // namespace op_api
