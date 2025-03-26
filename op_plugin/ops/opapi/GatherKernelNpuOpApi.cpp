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

at::Tensor& gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnGather, acl_op::gather_out(self, dim, index, sparse_grad, out));
    auto output_size = index.sizes();
    npu_preparation::check_tensor(
        {self},
        out,
        self.scalar_type(),
        output_size);
    EXEC_NPU_CMD(aclnnGather, self, dim, index, out);
    return out;
}

at::Tensor& gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnGather, acl_op::gather_out(self, dim, index, sparse_grad, out));
    auto output_size = index.sizes();
    npu_preparation::check_tensor(
        {self},
        out,
        self.scalar_type(),
        output_size);
    const int64_t real_dim = dimname_to_position(self, dim);
    EXEC_NPU_CMD(aclnnGather, self, real_dim, index, out);
    return out;
}

at::Tensor gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad)
{
    DO_COMPATIBILITY(aclnnGather, acl_op::gather(self, dim, index, sparse_grad));
    auto outputSize = index.sizes();
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
    EXEC_NPU_CMD(aclnnGather, self, dim, index, result);
    return result;
}

at::Tensor gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad)
{
    DO_COMPATIBILITY(aclnnGather, acl_op::gather(self, dim, index, sparse_grad));
    auto outputSize = index.sizes();
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
    const int64_t real_dim = dimname_to_position(self, dim);
    EXEC_NPU_CMD(aclnnGather, self, real_dim, index, result);
    return result;
}
}
