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

constexpr int FOREACH_NORM_MAX_TENSOR_COUNT = 24;

bool checkData(const at::TensorList self, double p, const at::Scalar& scalar)
{
    if (scalar.isIntegral(false)) {
        p = scalar.to<int64_t>();
    } else if (scalar.isFloatingPoint()) {
        p = scalar.to<double>();
    } else {
        TORCH_CHECK(false, "foreach_tensor_norm_npu expects scalar to be integer or float", OPS_ERROR(ErrCode::TYPE));
    }
    const bool has_int_or_complex = std::any_of(self.begin(), self.end(), [](const auto &t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, true) || at::isComplexType(scalar_type);
    });
    return !at::native::can_use_fast_route(self) || has_int_or_complex ||
           !(p == static_cast<double>(1) || p == static_cast<double>(2));
}

void _split_and_exec_npu_cmd_norm(at::TensorList tensors1, const at::Scalar& scalar, at::TensorList result_list)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = FOREACH_NORM_MAX_TENSOR_COUNT;   // tensorlist切分长度
    
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachNorm, tensors1, scalar, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachNorm, temp_tensors1, scalar, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachNorm, temp_tensors1, scalar, temp_result);
    }
}

std::vector<at::Tensor> _foreach_norm(const at::TensorList self, const at::Scalar& scalar)
{
    double p;
    at::native::check_foreach_api_restrictions(self);

    DO_COMPATIBILITY(aclnnForeachNorm, at::native::foreach_tensor_norm_slow(self, scalar));

    if (checkData(self, p, scalar)) {
        return at::native::foreach_tensor_norm_slow(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();
    TORCH_CHECK(scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float ||
                scalar_type == at::ScalarType::BFloat16, "input must be half, float or bfloat16", OPS_ERROR(ErrCode::TYPE));
    std::vector<at::Tensor> result(self.size());
    auto iterRes = result.data();
    int i = 0;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        iterRes[i] = npu_preparation::apply_tensor_without_format(1, tensor.options().dtype(scalar_type));
        i++;
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_norm(self, scalar, result_);
    return result;
}
}
