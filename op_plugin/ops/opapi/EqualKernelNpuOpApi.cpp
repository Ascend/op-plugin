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

bool equal(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnEqual, acl_op::equal(self, other));
    if (!at::namedinference::are_names_equal(
        self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
        return false;
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format({1}, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnEqual, self, other, result);
    return result.item().to<bool>();
}
}
