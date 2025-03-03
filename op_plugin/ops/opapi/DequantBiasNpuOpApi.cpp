// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include <torch/csrc/autograd/custom_function.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_dequant_bias(const at::Tensor& x,
                            const at::Tensor& weight_scale,
                            const c10::optional<at::Tensor>& activation_scale,
                            const c10::optional<at::Tensor>& bias,
                            c10::optional<at::ScalarType> output_dtype)
{
    at::ScalarType dst_type = c10::value_or_else(output_dtype, [] {return at::ScalarType::Half;});
    TORCH_CHECK(dst_type == at::ScalarType::Half || dst_type == at::ScalarType::BFloat16,
        "The dtype should be half or bfloat16", OPS_ERROR(ErrCode::PARAM));

    at::Tensor result = npu_preparation::apply_tensor_without_format(x.sizes(), x.options().dtype(dst_type));
    int64_t dtype_num = static_cast<int64_t>(npu_preparation::convert_to_acl_data_type(dst_type));

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnDequantBias, x, weight_scale, activation_scale, bias, dtype_num, result);
    return result;
}
} // namespace op_api