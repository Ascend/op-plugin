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

std::tuple<at::Tensor, at::Tensor> slogdet(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnSlogdet, acl_op::slogdet(self));
    // input dimension at least 2
    TORCH_CHECK(self.ndimension() >= 2, "Expected nonempty least 2D tensor, but got a tensor with sizes ", self.dim(),
                OPS_ERROR(ErrCode::PARAM));
    // calculate the output size
    auto outputSize = op_infer::array_to_small_vector(self.sizes());
    outputSize.erase(outputSize.end() - 2, outputSize.end());
    // construct the output tensor of the NPU
    at::Tensor sign = npu_preparation::apply_tensor(self, outputSize);
    at::Tensor log = npu_preparation::apply_tensor(self, outputSize);
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

    return std::tie(sign, log);
}

#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor, at::Tensor> linalg_slogdet(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnSlogdet, acl_op::linalg_slogdet(self));
    // input dimension at least 2
    TORCH_CHECK(self.ndimension() >= 2, "Expected nonempty least 2D tensor, but got a tensor with sizes ",
                self.dim(), OPS_ERROR(ErrCode::PARAM));
    // calculate the output size
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.erase(output_size.end() - 2, output_size.end());
    // construct the output tensor of the NPU
    at::Tensor sign = npu_preparation::apply_tensor(self, output_size);
    at::Tensor log = npu_preparation::apply_tensor(self, output_size);
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

    return std::tie(sign, log);
}

std::tuple<at::Tensor&, at::Tensor&> linalg_slogdet_out(const at::Tensor& self,
                                                        at::Tensor& sign,
                                                        at::Tensor& log)
{
    DO_COMPATIBILITY(aclnnSlogdet, acl_op::linalg_slogdet_out(self, sign, log));
    EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

    return std::tie(sign, log);
}
#endif

}
