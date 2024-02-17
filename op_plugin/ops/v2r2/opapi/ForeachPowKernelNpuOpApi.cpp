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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_pow(at::TensorList tensors1, at::TensorList tensors2)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
        return at::native::foreach_tensor_pow_list_kernel_slow(tensors1, tensors2);
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

    EXEC_NPU_CMD(aclnnForeachPowList, tensors1, tensors2, result_);
    return result;
}

void _foreach_pow_(at::TensorList tensors1, at::TensorList tensors2)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at::native::can_use_fast_route(tensors1, tensors2, true)) {
        return at::native::foreach_tensor_pow_list_kernel_slow_(tensors1, tensors2);
    }

    EXEC_NPU_CMD(aclnnForeachPowList, tensors1, tensors2, tensors1);
    return;
}

std::vector<at::Tensor> _foreach_pow(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    return at::native::foreach_tensor_pow_scalarlist_kernel_slow(tensors, scalars);
}

void _foreach_pow_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    return at::native::foreach_tensor_pow_scalarlist_kernel_slow_(tensors, scalars);
}

}  // namespace op_api
