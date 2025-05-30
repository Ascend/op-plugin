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

at::Tensor& __ior__(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceBitwiseOrScalar, acl_op::__ior__(self, other));
    DO_COMPATIBILITY(aclnnInplaceBitwiseOrTensor, acl_op::__ior__(self, other));

    if (!torch_npu::utils::is_npu(other)) {
        const at::Scalar other_value = other.item();
        EXEC_NPU_CMD(aclnnInplaceBitwiseOrScalar, self, other_value);
    } else {
        EXEC_NPU_CMD(aclnnInplaceBitwiseOrTensor, self, other);
    }
    return self;
}

at::Tensor& __ior__(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceBitwiseOrScalar, acl_op::__ior__(self, other));
    EXEC_NPU_CMD(aclnnInplaceBitwiseOrScalar, self, other);
    return self;
}
}
