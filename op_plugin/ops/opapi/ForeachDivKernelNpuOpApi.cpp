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

void _foreach_div_v1_(const at::TensorList self, const at::Scalar &scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_div_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float", OPS_ERROR(ErrCode::TYPE));
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type, self[0].device());
    EXEC_NPU_CMD(aclnnForeachDivScalar, self, scalar_tensor, self);
}

std::vector<at::Tensor> _foreach_div_v1(at::TensorList self, const at::Scalar &scalar)
{
    // Fallback
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_div_scalar_kernel_slow(self, scalar);
    }

    // Type Check
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float) {
        TORCH_CHECK(false, "input must be half or float", OPS_ERROR(ErrCode::TYPE));
    }

    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type, self[0].device());
    EXEC_NPU_CMD(aclnnForeachDivScalar, self, scalar_tensor, result_);

    return result;
}

void _split_and_exec_npu_cmd_div(at::TensorList &tensors1, at::TensorList &tensors2,
                                 at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;

    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
            EXEC_NPU_CMD(aclnnForeachDivList, tensors1, tensors2, result_list);
            return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachDivList, temp_tensors1, temp_tensors2, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachDivList, temp_tensors1, temp_tensors2, temp_result);
    }
}

void _split_and_exec_npu_cmd_div_scalar_list(at::TensorList& tensors1, at::ArrayRef<at::Scalar> scalars,
                                             at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 50 : 24;

    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
            EXEC_NPU_CMD(aclnnForeachDivScalarList, tensors1, scalars, result_list);
            return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachDivScalarList, temp_tensors1, temp_scalars, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachDivScalarList, temp_tensors1, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_div(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachDivList, at::native::foreach_tensor_div_list_kernel_slow(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_list_kernel_slow(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
        return at::native::foreach_tensor_div_list_kernel_slow(tensors1, tensors2);
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

    _split_and_exec_npu_cmd_div(tensors1, tensors2, result_, false);
    return result;
}

void _foreach_div_(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachDivList, at::native::foreach_tensor_div_list_kernel_slow_(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_list_kernel_slow_(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
        return at::native::foreach_tensor_div_list_kernel_slow_(tensors1, tensors2);
    }

    _split_and_exec_npu_cmd_div(tensors1, tensors2, tensors1, true);
    return;
}

std::vector<at::Tensor> _foreach_div(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachDivScalarList,
                     at::native::foreach_tensor_div_scalarlist_kernel_slow(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_scalarlist_kernel_slow(tensors, scalars);
    }

    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, true)) {
        return at::native::foreach_tensor_div_scalarlist_kernel_slow(tensors, scalars);
    }

    // Type Check
    auto scalar_type = tensors[0].scalar_type();
    if (scalar_type != at::ScalarType::Half
        && scalar_type != at::ScalarType::Float
        && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float or bfloat16");
    }

    std::vector<at::Tensor> result;

    result.reserve(tensors.size());
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_div_scalar_list(tensors, scalars, result_, false);

    return result;
}

void _foreach_div_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachDivScalarList,
                     at::native::foreach_tensor_div_scalarlist_kernel_slow_(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_scalarlist_kernel_slow_(tensors, scalars);
    }

    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, true)) {
        return at::native::foreach_tensor_div_scalarlist_kernel_slow_(tensors, scalars);
    }

    auto scalar_type = tensors[0].scalar_type();
    if (scalar_type != at::ScalarType::Half
        && scalar_type != at::ScalarType::Float
        && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float or bfloat16");
    }
    _split_and_exec_npu_cmd_div_scalar_list(tensors, scalars, tensors, true);
}

void _split_and_exec_npu_cmd_div_scalar(at::TensorList& tensors1, const at::Scalar& scalar,
                                        at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 50 : 24;

    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, tensors1);

    if (tensor_count <= max_tensor_count) {
            EXEC_NPU_CMD(aclnnForeachDivScalarV2, tensors1, scalar_, result_list);
            return;
        }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachDivScalarV2, temp_tensors1, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachDivScalarV2, temp_tensors1, scalar_, temp_result);
    }
}

void _foreach_div_(at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_scalar_kernel_slow_(self, scalar);
    }
    DO_COMPATIBILITY(aclnnForeachDivScalarV2, _foreach_div_v1_(self, scalar));
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_div_scalar_kernel_slow_(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half
        && scalar_type != at::ScalarType::Float
        && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float or bfloat16");
    }
    _split_and_exec_npu_cmd_div_scalar(self, scalar, self, true);
}

std::vector<at::Tensor> _foreach_div(at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_div_scalar_kernel_slow(self, scalar);
    }
    DO_COMPATIBILITY(aclnnForeachDivScalarV2, _foreach_div_v1(self, scalar));
    // Fallback
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, true)) {
        return at::native::foreach_tensor_div_scalar_kernel_slow(self, scalar);
    }

    // Type Check
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half
        && scalar_type != at::ScalarType::Float
        && scalar_type != at::ScalarType::BFloat16) {
        TORCH_CHECK(false, "input must be half, float or bfloat16");
    }

    std::vector<at::Tensor> result(self.size());
    auto iterRes = result.data();
    int i = 0;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        iterRes[i++] = npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_div_scalar(self, scalar, result_, false);

    return result;
}
}  // namespace op_api
