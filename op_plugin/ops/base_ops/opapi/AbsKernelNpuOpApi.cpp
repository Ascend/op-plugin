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

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, result));
    npu_preparation::check_tensor({self}, result, self);
    EXEC_NPU_CMD(aclnnAbs, self, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor abs(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self));

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAbs, self, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor& abs_(at::Tensor& self) {
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_(self));
    op_api::abs_out(self, self);
    return self;
}

}  // namespace op_api
