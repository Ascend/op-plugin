// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor scatter(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    acl_op::scatter_out(self, dim, index, src, result);
    return result;
}

at::Tensor scatter(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar &value)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    acl_op::scatter_out(self, dim, index, value, result);
    return result;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index_ex, const at::Tensor &src_ex)
{
    at::Tensor src(src_ex);
    scatter_npu_src_impl(self, dim, index_ex, src);
    return self;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index_ex, const at::Scalar &src)
{
    at::Tensor src_tensor = npu_preparation::copy_scalar_to_device(src, self.scalar_type());
    at::Tensor src_tensor_broadcast =
        acl_op::npu_broadcast(src_tensor, op_infer::array_to_small_vector(index_ex.sizes()));
    scatter_npu_src_impl(self, dim, index_ex, src_tensor_broadcast);
    return self;
}
} // namespace acl_op
