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
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void _split_and_exec_npu_cmd_lerp_scalar(at::TensorList &tensors1, at::TensorList &tensors2,
    const at::Scalar &weight, at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar weight_ = op_api::adaptToDouble(weight, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachLerpScalar, tensors1, tensors2, weight_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachLerpScalar, temp_tensors1, temp_tensors2, weight_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachLerpScalar, temp_tensors1, temp_tensors2, weight_, temp_result);
    }
}

void exec_npu_cmd_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar& weight)
{
    // dispatch hostAPI
    _split_and_exec_npu_cmd_lerp_scalar(tensors1, tensors2, weight, tensors1, true);
}

std::vector<at::Tensor> exec_npu_cmd(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar& weight)
{
    auto scalarType = tensors1[0].scalar_type();
    // construct the output tensorlist of the NPU
    std::vector<at::Tensor> result;
    for (size_t i = 0; i < tensors1.size(); i++) {
        at::Tensor tensor = tensors1[i];
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(at_npu::native::OpPreparation::apply_tensor_without_format(
            output_size, tensor.options().dtype(scalarType)));
    }

    at::TensorList result_ = at::TensorList(result);

    // dispatch hostAPI
    _split_and_exec_npu_cmd_lerp_scalar(tensors1, tensors2, weight, result_, false);
    return result;
}

void _foreach_lerp_(const at::TensorList tensors1, const at::TensorList tensors2, const at::Scalar& weight)
{
    DO_COMPATIBILITY(aclnnForeachLerpScalar,
                     at::native::foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route({tensors1, tensors2}, weight)) {
        return at::native::foreach_tensor_lerp_list_kernel_slow_(tensors1, tensors2, weight);
    }
    exec_npu_cmd_(tensors1, tensors2, weight);
}

std::vector<at::Tensor> _foreach_lerp(const at::TensorList tensors1,
                                      const at::TensorList tensors2,
                                      const at::Scalar& weight)
{
    DO_COMPATIBILITY(aclnnForeachLerpScalar,
                     at::native::foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
    }
    
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route({tensors1, tensors2}, weight)) {
        return at::native::foreach_tensor_lerp_list_kernel_slow(tensors1, tensors2, weight);
    }
    return exec_npu_cmd(tensors1, tensors2, weight);
}
#endif
} // namespace op_api
