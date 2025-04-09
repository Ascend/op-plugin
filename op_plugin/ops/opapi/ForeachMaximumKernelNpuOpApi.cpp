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
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachMaximumList, at::native::foreach_tensor_maximum_slow(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_maximum_slow(tensors1, tensors2);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors1[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_maximum_slow(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, false)) {
        return at::native::foreach_tensor_maximum_slow(tensors1, tensors2);
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

    EXEC_NPU_CMD(aclnnForeachMaximumList, tensors1, tensors2, result_);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachMaximumList,
                     at::native::foreach_tensor_clamp_max_list_kernel_slow(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_max_list_kernel_slow(tensors1, tensors2);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors1[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_clamp_max_list_kernel_slow(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, false)) {
        return at::native::foreach_tensor_clamp_max_list_kernel_slow(tensors1, tensors2);
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

    EXEC_NPU_CMD(aclnnForeachMaximumList, tensors1, tensors2, result_);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
std::vector<at::Tensor> _foreach_maximum_v1(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow(tensors, scalar);
    }
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(scalar, scalar_type, tensors[0].device());
    EXEC_NPU_CMD(aclnnForeachMaximumScalar, tensors, scalar_, result_);
    return result;
    }

void _foreach_maximum_v1_(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow_(tensors, scalar);
    }
    auto scalar_type = tensors[0].scalar_type();
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(scalar, scalar_type, tensors[0].device());
    EXEC_NPU_CMD(aclnnForeachMaximumScalar, tensors, scalar_, tensors);
    return;
}

void _split_and_exec_npu_cmd_max(at::TensorList& tensors1, at::TensorList& tensors2,
                                 at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;

    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMaximumList, tensors1, tensors2, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMaximumList, temp_tensors1, temp_tensors2, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMaximumList, temp_tensors1, temp_tensors2, temp_result);
    }
}

std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachMaximumList,
                     at::native::foreach_tensor_clamp_min_list_kernel_slow(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow(tensors1, tensors2);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors1[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, false)) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow(tensors1, tensors2);
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

    _split_and_exec_npu_cmd_max(tensors1, tensors2, result_, false);
    return result;
}

void _foreach_maximum_(at::TensorList tensors1, at::TensorList tensors2)
{
    DO_COMPATIBILITY(aclnnForeachMaximumList,
                     at::native::foreach_tensor_clamp_min_list_kernel_slow_(tensors1, tensors2));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow_(tensors1, tensors2);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors1[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow_(tensors1, tensors2);
    }

    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, false)) {
        return at::native::foreach_tensor_clamp_min_list_kernel_slow_(tensors1, tensors2);
    }

    _split_and_exec_npu_cmd_max(tensors1, tensors2, tensors1, true);
    return;
}

void _split_and_exec_npu_cmd_max_scalar(at::TensorList& tensors1, const at::Scalar& scalar,
                                        at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;

    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMaximumScalarV2, tensors1, scalar_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMaximumScalarV2, temp_tensors1, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMaximumScalarV2, temp_tensors1, scalar_, temp_result);
    }
}

void _split_and_exec_npu_cmd_max_scalar_list(at::TensorList& tensors1, at::ArrayRef<at::Scalar> scalars,
                                             at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMaximumScalarList, tensors1, scalars, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMaximumScalarList, temp_tensors1, temp_scalars, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMaximumScalarList, temp_tensors1, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);

    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow(tensors, scalar);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_SCALAR, scalar.type(),
                                               op_plugin::utils::ForeachMappingType::MAP_SCALAR_DEFAULT)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow(tensors, scalar);
    }

    DO_COMPATIBILITY(aclnnForeachMaximumScalarV2, _foreach_maximum_v1(tensors, scalar));
    if (!at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow(tensors, scalar);
    }
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_max_scalar(tensors, scalar, result_, false);
    return result;
}

void _foreach_maximum_(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);

    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow_(tensors, scalar);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_SCALAR, scalar.type(),
                                               op_plugin::utils::ForeachMappingType::MAP_SCALAR_DEFAULT)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow_(tensors, scalar);
    }

    DO_COMPATIBILITY(aclnnForeachMaximumScalarV2, _foreach_maximum_v1_(tensors, scalar));
    if (!at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_min_scalar_kernel_slow_(tensors, scalar);
    }

    _split_and_exec_npu_cmd_max_scalar(tensors, scalar, tensors, true);
    return;
}

std::vector<at::Tensor> _foreach_maximum(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachMaximumScalarList,
                     at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow(tensors, scalars);
    }

    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, false)) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow(tensors, scalars);
    }
    
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors[0].scalar_type();
    if (!op_plugin::utils::check_dtype_foreach(tensors[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_SCALARLIST, scalars[0].type(),
                                               op_plugin::utils::ForeachMappingType::MAP_SCALARLIST_DEFAULT)) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow(tensors, scalars);
    }

    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_max_scalar_list(tensors, scalars, result_, false);
    return result;
}

void _foreach_maximum_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    DO_COMPATIBILITY(aclnnForeachMaximumScalarList,
                     at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow_(tensors, scalars));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow_(tensors, scalars);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32,
                                               op_plugin::utils::ForeachInputType::TYPE_SCALARLIST, scalars[0].type(),
                                               op_plugin::utils::ForeachMappingType::MAP_SCALARLIST_DEFAULT)) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow_(tensors, scalars);
    }

    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, false)) {
        return at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow_(tensors, scalars);
    }

    _split_and_exec_npu_cmd_max_scalar_list(tensors, scalars, tensors, true);
    return;
}

std::vector<at::Tensor> _foreach_clamp_min(at::TensorList tensors1, at::TensorList tensors2)
{
    return op_api::_foreach_maximum(tensors1, tensors2);
}

void _foreach_clamp_min_(at::TensorList tensors1, at::TensorList tensors2)
{
    op_api::_foreach_maximum_(tensors1, tensors2);
    return;
}

std::vector<at::Tensor> _foreach_clamp_min(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    return op_api::_foreach_maximum(tensors, scalars);
}

void _foreach_clamp_min_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    op_api::_foreach_maximum_(tensors, scalars);
    return;
}

std::vector<at::Tensor> _foreach_clamp_min(at::TensorList tensors, const at::Scalar& scalar)
{
    return op_api::_foreach_maximum(tensors, scalar);
}

void _foreach_clamp_min_(at::TensorList tensors, const at::Scalar& scalar)
{
    op_api::_foreach_maximum_(tensors, scalar);
    return;
}
#endif
}
