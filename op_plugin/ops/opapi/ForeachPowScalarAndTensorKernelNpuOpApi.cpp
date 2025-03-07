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
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void _split_and_exec_npu_cmd_pow_scalar(const at::Scalar& scalar, const at::TensorList exponent,
                                        at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = exponent.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, exponent);
    scalar_ = op_api::adaptToInteger(scalar_, exponent);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachPowScalarAndTensor, scalar_, exponent, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_exponent(exponent.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachPowScalarAndTensor, scalar_, temp_exponent, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_exponent(exponent.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachPowScalarAndTensor, scalar_, temp_exponent, temp_result);
    }
}

std::vector<at::Tensor> _foreach_pow(const at::Scalar& scalar, const at::TensorList exponent)
{
    DO_COMPATIBILITY(aclnnForeachPowScalarAndTensor, at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(exponent[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_SCALAR, scalar.type(),
                                               op_plugin::utils::ForeachMappingType::MAP_POW_SCALAR_AND_TENSOR)) {
        return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
    }

    at::native::check_foreach_api_restrictions(exponent);
    if (!at::native::can_use_fast_route(exponent, scalar, true)) {
        return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
    }

    auto scalar_type = exponent[0].scalar_type();

    std::vector<at::Tensor> result;
    result.reserve(exponent.size());
    for (const at::Tensor &tensor : exponent) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }

    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_pow_scalar(scalar, exponent, result_, false);
    return result;
}
#endif
}
