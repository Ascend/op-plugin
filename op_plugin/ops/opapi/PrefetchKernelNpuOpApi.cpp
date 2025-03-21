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
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
void npu_prefetch(const at::Tensor &self,
                  const c10::optional<at::Tensor> &dependency,
                  int64_t max_size,
                  int64_t offset)
{
    TORCH_CHECK(max_size > 0, "max_size should be greater than zero, but got ", max_size,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(offset >= 0, "offset should not be smaller than zero, but got ", offset,
                OPS_ERROR(ErrCode::PARAM));

    auto dtype = c10::scalarTypeToTypeMeta(self.scalar_type());
    int64_t nelements = 0;
    if (at_npu::native::FormatHelper::IsBaseFormatType(self)) {
        nelements = c10::multiply_integers(self.sizes());
    } else {
        nelements = c10::multiply_integers(torch_npu::NPUBridge::GetNpuStorageImplDesc(self).storage_sizes_);
    }
    int64_t tensor_size = static_cast<int64_t>(dtype.itemsize()) * nelements;

    TORCH_CHECK(
        tensor_size > offset,
        "offset out of range of tensor size, tensor size: ",
        tensor_size,
        ", offset: ",
        offset,
        OPS_ERROR(ErrCode::PARAM));
    if ((tensor_size - offset) < max_size) {
        max_size = tensor_size - offset;
    }
    aclrtStream current_stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR_WITHOUT_UCE(
        c10_npu::acl::AclrtCmoAsync((char*)self.data_ptr() + offset,
        max_size,
        ACL_RT_CMO_TYPE_PREFETCH,
        current_stream));
}
#endif
} // namespace op_api