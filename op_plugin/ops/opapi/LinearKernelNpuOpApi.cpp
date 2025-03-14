// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_linear(
    const at::Tensor &input,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias)
{
    const at::Tensor &bias_opt = bias.value_or(at::Tensor());
    const at::Tensor &weight_t = weight.t();
    auto output_size = {input.size(0), weight.size(0)};
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, input.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());

    if (bias_opt.defined()) {
        const at::Scalar beta = 1;
        const at::Scalar alpha = 1;
        DO_COMPATIBILITY(aclnnAddmm, acl_op::addmm(bias_opt, input, weight_t, beta, alpha));
        EXEC_NPU_CMD(aclnnAddmm, bias_opt, input, weight_t, beta, alpha, result, cube_math_type);
        return result;
    }
    DO_COMPATIBILITY(aclnnMm, acl_op::mm(input, weight_t));
    EXEC_NPU_CMD(aclnnMm, input, weight_t, result, cube_math_type);
    return result;
}

std::tuple<at::Tensor, at::Tensor> npu_linear_backward(
    const at::Tensor &grad,
    const at::Tensor &input,
    const at::Tensor &weight)
{
    DO_COMPATIBILITY(aclnnMm, acl_op::npu_linear_backward(grad, input, weight));
    at::Tensor input_grad = npu_preparation::apply_tensor_without_format(input.sizes(), grad.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMm, grad, weight, input_grad, cube_math_type);

    const at::Tensor &grad_t = grad.t();
    at::Tensor weight_grad = npu_preparation::apply_tensor_without_format(weight.sizes(), grad.options());
    EXEC_NPU_CMD(aclnnMm, grad_t, input, weight_grad, cube_math_type);

    return std::tie(input_grad, weight_grad);
}

}
