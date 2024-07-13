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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _split_and_exec_npu_cmd_min(at::TensorList& tensors1, at::TensorList& tensors2, at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMinimumList, tensors1, tensors2, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMinimumList, temp_tensors1, temp_tensors2, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMinimumList, temp_tensors1, temp_tensors2, temp_result);
    }
}

std::vector<at::Tensor> _foreach_minimum(at::TensorList tensors1, at::TensorList tensors2)
{
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

    _split_and_exec_npu_cmd_min(tensors1, tensors2, result_, false);
    return result;
}

void _foreach_minimum_(at::TensorList tensors1, at::TensorList tensors2)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, false)) {
        return at::native::foreach_tensor_clamp_max_list_kernel_slow_(tensors1, tensors2);
    }

    _split_and_exec_npu_cmd_min(tensors1, tensors2, tensors1, true);
    return;
}

void _split_and_exec_npu_cmd_min_scalar(at::TensorList& tensors1, const at::Scalar& scalar, at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 50 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMinimumScalar, tensors1, scalar_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMinimumScalar, temp_tensors1, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMinimumScalar, temp_tensors1, scalar_, temp_result);
    }
}

void _split_and_exec_npu_cmd_min_scalar_list(at::TensorList& tensors1, at::ArrayRef<at::Scalar> scalars, at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMinimumScalarList, tensors1, scalars, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMinimumScalarList, temp_tensors1, temp_scalars, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMinimumScalarList, temp_tensors1, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_minimum(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_max_scalar_kernel_slow(tensors, scalar);
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
    _split_and_exec_npu_cmd_min_scalar(tensors, scalar, result_, false);
    return result;
    }

void _foreach_minimum_(at::TensorList tensors, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors, scalar, false)) {
        return at::native::foreach_tensor_clamp_max_scalar_kernel_slow_(tensors, scalar);
    }
    
    _split_and_exec_npu_cmd_min_scalar(tensors, scalar, tensors, true);
    return;
}

std::vector<at::Tensor> _foreach_minimum(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, false)) {
        return at::native::foreach_tensor_clamp_max_scalarlist_kernel_slow(tensors, scalars);
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
    _split_and_exec_npu_cmd_min_scalar_list(tensors, scalars, result_, false);
    return result;
}

void _foreach_minimum_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(tensors, scalars);
    if (!at::native::can_use_fast_route(tensors, scalars, false)) {
        return at::native::foreach_tensor_clamp_max_scalarlist_kernel_slow_(tensors, scalars);
    }
    
    _split_and_exec_npu_cmd_min_scalar_list(tensors, scalars, tensors, true);
    return;
}

}  // namespace op_api
