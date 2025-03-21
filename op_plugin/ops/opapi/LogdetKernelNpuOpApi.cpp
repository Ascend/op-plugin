// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor logdet(const at::Tensor &self)
{
    DO_COMPATIBILITY(aclnnLogdet, acl_op::logdet(self));
    // input dimension at least 2
    TORCH_CHECK(
        self.ndimension() >= 2,
        "Expected nonempty least 2D tensor, but got a tensor with sizes ",
        self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    // calculate the output size
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.erase(output_size.end() - 2, output_size.end());
    // construct the output tensor of the NPU
    at::Tensor log = npu_preparation::apply_tensor(self, output_size);
    EXEC_NPU_CMD(aclnnLogdet, self, log);

    return log;
}
#endif

} // namespace op_api
