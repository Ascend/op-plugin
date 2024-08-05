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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(tensors) ||
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
    EXEC_NPU_CMD(aclnnForeachSqrt, tensors, result_);
    return result;
}

void _foreach_sqrt_(at::TensorList tensors)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(tensors) ||
        at::native::has_integral_tensor(tensors, true)) {
        return at::native::foreach_tensor_sqrt_slow_(tensors);
    }
    EXEC_NPU_CMD(aclnnForeachSqrt, tensors, tensors);
    return;
}

}
