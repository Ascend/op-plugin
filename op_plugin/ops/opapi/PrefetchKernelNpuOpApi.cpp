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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void npu_prefetch(const at::Tensor &self,
                  const c10::optional<at::Tensor> &dependency,
                  int64_t max_size)
{
    TORCH_CHECK(max_size > 0, "kernel size should be greater than zero, but got ", max_size,
                OPS_ERROR(ErrCode::PARAM));
    
    int64_t tensor_size = static_cast<int64_t>(elementSize(self.scalar_type()));
    for (int64_t index = 0; index < self.dim(); index++) {
        tensor_size *= self.size(index);
    }
    if (tensor_size < max_size) {
        max_size = tensor_size;
    }
    aclrtStream current_stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtCmoAsync(self.data_ptr(), max_size, ACL_RT_CMO_TYPE_PREFETCH, current_stream));
}
#endif
} // namespace op_api