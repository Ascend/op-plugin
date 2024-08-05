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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

void _split_and_exec_npu_cmd_atan(at::TensorList tensors1, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachAtan, tensors1, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachAtan, temp_tensors1, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachAtan, temp_tensors1, temp_result);
    }
}

void _foreach_atan_(const at::TensorList self_atan)
{
    at::native::check_foreach_api_restrictions(self_atan);
    if (!at::native::can_use_fast_route(self_atan) || at::native::has_integral_tensor(self_atan, true)) {
        return at::native::foreach_tensor_atan_slow_(self_atan);
    }

    if (self_atan.empty()) {
        return;
    }
    
    _split_and_exec_npu_cmd_atan(self_atan, self_atan, true);
}


std::vector<at::Tensor> _foreach_atan(const at::TensorList self_atan)
{
    at::native::check_foreach_api_restrictions(self_atan);
    if (!at::native::can_use_fast_route(self_atan) || at::native::has_integral_tensor(self_atan, true)) {
        return at::native::foreach_tensor_atan_slow(self_atan);
    }

    auto scalar_type = self_atan[0].scalar_type();

    // construct output tensorlist
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self_atan) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_atan(self_atan, result_, false);
    return result;
}
} // namespace at_npu
