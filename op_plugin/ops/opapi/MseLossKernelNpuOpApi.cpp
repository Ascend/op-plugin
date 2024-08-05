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
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& mse_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
    DO_COMPATIBILITY(aclnnMseLossOut, acl_op::mse_loss_out(self, target, reduction, result));
    auto output_size = op_infer::mse_loss_npu_output_size(self, target, reduction);
    at_npu::native::OpPreparation::check_tensor({self, target}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnMseLossOut, self, target, reduction, result);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor& mse_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
    DO_COMPATIBILITY(aclnnMseLoss, acl_op::mse_loss_out(self, target, reduction, result));
    at::IntArrayRef output_size;
    if (reduction == at::Reduction::None) {
        output_size = op_infer::broadcast_ops_npu_output_size(self, target);
    }
    at_npu::native::OpPreparation::check_tensor({self, target}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnMseLoss, self, target, reduction, result);
    return result;
}
#endif

at::Tensor mse_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
    DO_COMPATIBILITY(aclnnMseLoss, acl_op::mse_loss(self, target, reduction));
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    if (reduction == at::Reduction::None) {
        output_size = op_infer::broadcast_ops_npu_output_size(self, target);
    }
    at::ScalarType high_type = at::native::result_type(self, target);
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(high_type));
    EXEC_NPU_CMD(aclnnMseLoss, self, target, reduction, result);
    return result;
}

}
