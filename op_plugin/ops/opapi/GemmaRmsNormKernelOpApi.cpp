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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& self,
    const at::Tensor& gamma,
    double epsilon)
{
    auto output_size = op_infer::rms_norm_npu_output_size(self, gamma);
    at::Tensor y = npu_preparation::apply_tensor_with_format(output_size[0], self.options(), ACL_FORMAT_ND);
    at::Tensor rstd = npu_preparation::apply_tensor_with_format(output_size[1], self.options().dtype(at::kFloat), ACL_FORMAT_ND);
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, self, gamma, epsilon, y, rstd);
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

}