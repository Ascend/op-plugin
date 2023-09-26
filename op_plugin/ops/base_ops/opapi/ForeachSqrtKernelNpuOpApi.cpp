// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors)
{
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
    EXEC_NPU_CMD(aclnnForeachSqrt, tensors, result_);
    return result;
}

void _foreach_sqrt_(at::TensorList tensors)
{
    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors) ||
        at::native::has_integral_tensor(tensors, true)) {
        return at::native::foreach_tensor_sqrt_slow_(tensors);
    }
    EXEC_NPU_CMD(aclnnForeachSqrt, tensors, tensors);
    return;
}

}  // namespace op_api
