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

at::Tensor& index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& result) {
    DO_COMPATIBILITY(aclnnIndexAdd, acl_op::index_add_out(self, dim, index, source, alpha, result));
    at_npu::native::OpPreparation::check_tensor({self, index, source},
                                                result,
                                                result.scalar_type(),
                                                self.sizes());
    if (!result.is_same(self)) {
        result.copy_(self);
    }
    EXEC_NPU_CMD(aclnnIndexAdd, result, dim, index, source, alpha, result);
    return result;
}

at::Tensor index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
    DO_COMPATIBILITY(aclnnIndexAdd, acl_op::index_add(self, dim, index, source, alpha));
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());
    EXEC_NPU_CMD(aclnnIndexAdd, result.copy_(self), dim, index, source, alpha, result.copy_(self));
    return result;
}

}
