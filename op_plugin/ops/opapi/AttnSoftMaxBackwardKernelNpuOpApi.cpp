// Copyright (c) 2025 Huawei Technologies Co., Ltd
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


namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &npu_attn_softmax_backward_(at::Tensor &self, const at::Tensor &grad_output, const at::Tensor &values)
{
    // allow dicrease precision
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    at::Tensor values_tmp = values;
    values_tmp = values_tmp.transpose(-2, -1);
    auto output_size = op_infer::matmul_output_size(grad_output, values_tmp);
    auto matmul_result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, grad_output.options());
    EXEC_NPU_CMD(aclnnMatmul, grad_output, values_tmp, matmul_result, cube_math_type);

    int64_t dim = -1;
    EXEC_NPU_CMD(aclnnSoftmaxBackward, matmul_result, self, dim, self);
    return self;
}
}
