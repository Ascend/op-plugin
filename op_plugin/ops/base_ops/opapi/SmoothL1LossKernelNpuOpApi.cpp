// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

at::Tensor &smooth_l1_loss_out(const at::Tensor &self, const at::Tensor &target, int64_t reduction, double beta,
                               at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnSmoothL1Loss, acl_op::smooth_l1_loss_out(self, target, reduction, beta, result));
    auto outputSize = op_infer::smooth_l1_loss_npu_output_size(self, target, reduction);
    npu_preparation::check_tensor({self, target}, result, result.scalar_type(), outputSize);
    npu_preparation::check_memory({self, target}, {result});
    float sigma = static_cast<float>(beta);
    EXEC_NPU_CMD(aclnnSmoothL1Loss, self, target, reduction, sigma, result);
    return result;
}

at::Tensor smooth_l1_loss(const at::Tensor &self, const at::Tensor &target, int64_t reduction, double beta)
{
    DO_COMPATIBILITY(aclnnSmoothL1Loss, acl_op::smooth_l1_loss(self, target, reduction, beta));
    auto outputSize = op_infer::smooth_l1_loss_npu_output_size(self, target, reduction);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
    float sigma = static_cast<float>(beta);
    EXEC_NPU_CMD(aclnnSmoothL1Loss, self, target, reduction, sigma, result);
    return result;
}
} // namespace op_api
