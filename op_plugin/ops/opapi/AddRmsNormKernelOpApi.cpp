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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon)
{
    DO_COMPATIBILITY(aclnnAddRmsNorm, acl_op::npu_add_rms_norm(x1, x2, gamma, epsilon));
    auto output_size = op_infer::rms_norm_npu_output_size(x1, gamma);
    at::Tensor y = npu_preparation::apply_tensor_without_format(output_size[0], x1.options());
    at::Tensor rstd = npu_preparation::apply_tensor_without_format(output_size[1], x1.options().dtype(at::kFloat));
    at::Tensor x = npu_preparation::apply_tensor_without_format(output_size[0], x1.options());

    EXEC_NPU_CMD(aclnnAddRmsNorm, x1, x2, gamma, epsilon, y, rstd, x);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, rstd, x);
}

}
