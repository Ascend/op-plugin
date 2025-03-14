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

at::Tensor& batch_norm_elemt_out(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
                                 const c10::optional<at::Tensor>& bias, const at::Tensor& mean,
                                 const at::Tensor& invstd, double eps, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnBatchNormElemt,
                     acl_op::batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out));
    const at::Tensor& bias_value = c10::value_or_else(bias, [] { return at::Tensor(); });
    const at::Tensor& weight_value = c10::value_or_else(weight, [] { return at::Tensor(); });
    npu_preparation::check_tensor({input, weight_value, bias_value, mean, invstd}, out, input);
    EXEC_NPU_CMD(aclnnBatchNormElemt, input, weight_value, bias_value, mean, invstd, eps, out);
    return out;
}

at::Tensor batch_norm_elemt(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
                            const c10::optional<at::Tensor>& bias, const at::Tensor& mean, const at::Tensor& invstd,
                            double eps)
{
    DO_COMPATIBILITY(aclnnBatchNormElemt, acl_op::batch_norm_elemt(input, weight, bias, mean, invstd, eps));
    at::Tensor result = npu_preparation::apply_tensor_without_format(input);
    EXEC_NPU_CMD(aclnnBatchNormElemt, input, weight, bias, mean, invstd, eps, result);
    return result;
}

}
