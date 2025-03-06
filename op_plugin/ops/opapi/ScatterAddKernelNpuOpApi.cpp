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

at::Tensor scatter_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src)
{
    DO_COMPATIBILITY(aclnnScatterAdd, acl_op::scatter_add(self, dim, index, src));
    auto selfClone = self.clone(at::MemoryFormat::Contiguous);
    npu_preparation::CheckMemory({selfClone, index, src}, {selfClone});
    EXEC_NPU_CMD(aclnnScatterAdd, selfClone, dim, index, src, selfClone);
    return selfClone;
}

at::Tensor& scatter_add_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src)
{
    DO_COMPATIBILITY(aclnnScatterAdd, acl_op::scatter_add_(self, dim, index, src));
    npu_preparation::CheckMemory({self, index, src}, {self});
    EXEC_NPU_CMD(aclnnScatterAdd, self, dim, index, src, self);
    return self;
}

at::Tensor scatter_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& src)
{
    DO_COMPATIBILITY(aclnnScatterAdd, acl_op::scatter_add(self, dim, index, src));
    return op_api::scatter_add(self, dimname_to_position(self, dim), index, src);
}

}
