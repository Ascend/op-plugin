// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Utils.h"

namespace atb {
namespace utils {

aclDataType ConvertToAclDataType(const at::ScalarType &data_type)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
                std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

at::Tensor FormatTrans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

}  // namespace utils
}  // namespace atb
