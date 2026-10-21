// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

// A2/A3 等支持 acl_op 的老 SOC 保持原有 acl_op(GE AsStrided) 实现，行为不变；
// 仅 Ascend950(A5) 等 aclnn-only 设备在 ATen 层用 as_strided 构造纯元数据视图，
// 再通过 aclnnInplaceCopy 将其物化到连续输出上。
at::Tensor npu_stride_copy(
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset) {
  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_stride_copy(self, shape, stride, storage_offset);
  }
  DO_COMPATIBILITY(aclnnInplaceCopy, acl_op::npu_stride_copy(self, shape, stride, storage_offset));
  at::Tensor view = self.as_strided(shape, stride, storage_offset.toLong());
  at::Tensor result = npu_preparation::apply_tensor_without_format(shape, self.options());
  EXEC_NPU_CMD(aclnnInplaceCopy, result, view);
  return result;
}

at::Tensor& npu_stride_copy_out(
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset,
    at::Tensor& out) {
  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out);
  }
  DO_COMPATIBILITY(aclnnInplaceCopy, acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out));
  npu_preparation::check_tensor({self}, out, shape);
  at::Tensor view = self.as_strided(shape, stride, storage_offset.toLong());
  EXEC_NPU_CMD(aclnnInplaceCopy, out, view);
  return out;
}

} // namespace op_api
