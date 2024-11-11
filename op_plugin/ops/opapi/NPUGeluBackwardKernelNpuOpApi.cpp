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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_gelu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate)
{
    std::string approximate_str = std::string(approximate);
    TORCH_CHECK(approximate_str == "tanh" || approximate_str == "none",
        "NPU error, approximate argument must be either none or tanh.", OPS_ERROR(ErrCode::PARAM));
    
    at::ScalarType high_type = at::native::result_type(grad_output, self);
    auto output_size = op_infer::broadcast_ops_npu_output_size(grad_output, self);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(high_type));

    char *approximate_ptr = const_cast<char *>(approximate_str.c_str());
    EXEC_NPU_CMD(aclnnGeluBackwardV2, grad_output, self, approximate_ptr, result);
    return result;
}
}
