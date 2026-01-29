// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
#if VERSION_BETWEEN(V1R11, V2R5)
at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate)
{
    DO_COMPATIBILITY(aclnnGeluBackward, acl_op::gelu_backward(grad_output, self, approximate));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                         grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGeluBackward, grad_output, self, grad_input);
    return grad_input;
}
#endif

#if VERSION_BETWEEN(V2R6, VERSION_NEWEST)
at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate)
{
    static auto gelu_sc = at_npu::native::env::CheckStrongConsistency();
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                         grad_output.options().dtype(output_dtype_0));
    auto approximate_str = op_infer::npu_gelu_approximate_str(approximate);
    auto approximate_ptr = const_cast<char *>(approximate_str.c_str());
    if (!gelu_sc) {
        DO_COMPATIBILITY(aclnnGeluBackward, acl_op::gelu_backward(grad_output, self, approximate));
        EXEC_NPU_CMD(aclnnGeluBackward, grad_output, self, grad_input);
    } else {
        EXEC_NPU_CMD(aclnnGeluBackwardV2, grad_output, self, approximate_ptr, grad_input);
    }
    return grad_input;
}
#endif
}
