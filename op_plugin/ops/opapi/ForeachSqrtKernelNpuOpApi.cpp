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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _split_and_exec_npu_cmd_sqrt(at::TensorList& tensors1, at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachSqrt, tensors1, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachSqrt, temp_tensors1, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachSqrt, temp_tensors1, temp_result);
    }
}

std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors)
{
    DO_COMPATIBILITY(aclnnForeachSqrt, at::native::foreach_tensor_sqrt_slow(tensors));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sqrt_slow(tensors);
    }

    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors) ||
        at::native::has_integral_tensor(tensors, true)) {
        return at::native::foreach_tensor_sqrt_slow(tensors);
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
    _split_and_exec_npu_cmd_sqrt(tensors, result_, false);
    return result;
}

void _foreach_sqrt_(at::TensorList tensors)
{
    DO_COMPATIBILITY(aclnnForeachSqrt, at::native::foreach_tensor_sqrt_slow_(tensors));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sqrt_slow_(tensors);
    }

    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors) ||
        at::native::has_integral_tensor(tensors, true)) {
        return at::native::foreach_tensor_sqrt_slow_(tensors);
    }
    _split_and_exec_npu_cmd_sqrt(tensors, tensors, true);
    return;
}
}  // namespace op_api