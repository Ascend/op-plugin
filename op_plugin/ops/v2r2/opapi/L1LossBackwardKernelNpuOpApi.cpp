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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor l1_loss_backward(const at::Tensor &grad_output, const at::Tensor &self, const at::Tensor &target,
                            int64_t reduction)
{
    DO_COMPATIBILITY(aclnnL1LossBackward, acl_op::l1_loss_backward(grad_output, self, target, reduction));
    // construct the output tensor of NPU
    auto output_size1_vec = op_infer::broadcast_ops_npu_output_size(self, target);
    at::IntArrayRef output_size1 = output_size1_vec;
    auto output_size2_vec = op_infer::broadcast_ops_npu_output_size(output_size1, grad_output.sizes());
    at::IntArrayRef output_size2 = output_size2_vec;
    // dtype promotion
    auto promote1 = at::native::result_type(target, self);
    auto grad_input_dtype = promoteTypes(grad_output.scalar_type(), promote1);
    // construct the output tensor of the NPU
    at::Tensor grad_input =
        npu_preparation::apply_tensor_without_format(output_size2, self.options().dtype(grad_input_dtype));
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnL1LossBackward, grad_output, self, target, reduction, grad_input);
    return grad_input;
}

} // namespace op_api
