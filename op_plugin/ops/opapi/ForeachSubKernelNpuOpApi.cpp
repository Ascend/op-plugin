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

std::vector<at::Tensor> _foreach_sub_v1(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar &alpha)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow(tensors1, tensors2, alpha);
    }
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors1[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors1) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    // convert scalar to tensor in PTA for now，wait for ascendc aclnn framwork support scalar type
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(alpha, scalar_type, tensors1[0].device());

    EXEC_NPU_CMD(aclnnForeachSubList, tensors1, tensors2, scalar_, result_);
    return result;
}

void _foreach_sub_v1_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar &alpha)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow_(tensors1, tensors2, alpha);
    }
    // convert scalar to tensor in PTA for now，wait for ascendc aclnn framwork support scalar type
    auto scalar_type = tensors1[0].scalar_type();
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(alpha, scalar_type, tensors1[0].device());

    EXEC_NPU_CMD(aclnnForeachSubList, tensors1, tensors2, scalar_, tensors1);
    return;
}

void _split_and_exec_npu_cmd_sub(at::TensorList &tensors1, at::TensorList &tensors2,
    const at::Scalar &alpha, at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;
    size_t loop_time = tensor_count / max_tensor_count;
    at::Scalar alpha_ = op_api::adaptToDouble(alpha, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachSubListV2, tensors1, tensors2, alpha_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachSubListV2, temp_tensors1, temp_tensors2, alpha_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachSubListV2, temp_tensors1, temp_tensors2, alpha_, temp_result);
    }
}

std::vector<at::Tensor> _foreach_sub(at::TensorList tensors1,
    at::TensorList tensors2, const at::Scalar &alpha)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_list_kernel_slow(tensors1, tensors2, alpha);
    }
    DO_COMPATIBILITY(aclnnForeachSubListV2, _foreach_sub_v1(tensors1, tensors2, alpha));
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow(tensors1, tensors2, alpha);
    }
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors1[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors1) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
            tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_sub(tensors1, tensors2, alpha, result_, false);
    return result;
}

void _foreach_sub_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar& alpha)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_list_kernel_slow_(tensors1, tensors2, alpha);
    }
    DO_COMPATIBILITY(aclnnForeachSubListV2, _foreach_sub_v1_(tensors1, tensors2, alpha));
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow_(tensors1, tensors2, alpha);
    }

    _split_and_exec_npu_cmd_sub(tensors1, tensors2, alpha, tensors1, true);
    return;
}

void _split_and_exec_npu_cmd_sub_scalarlist(at::TensorList& tensors1, at::ArrayRef<at::Scalar> scalars,
                                            at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;

    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachSubScalarList, tensors1, scalars, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachSubScalarList, temp_tensors1, temp_scalars, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachSubScalarList, temp_tensors1, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachSubScalarList, at::native::foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars);
    }

    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, true)) {
        return at::native::foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars);
    }

    auto scalar_type = tensors[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_sub_scalarlist(tensors, scalars, result_, false);
    return result;
}

void _foreach_sub_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachSubScalarList, at::native::foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars);
    }

    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, true)) {
        at::native::foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars);
        return;
    }

    _split_and_exec_npu_cmd_sub_scalarlist(tensors, scalars, tensors, true);
}
}  // namespace op_api
