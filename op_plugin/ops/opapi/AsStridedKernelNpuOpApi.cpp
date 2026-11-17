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
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// aclnn's storageOffset is view-relative (the GM address already carries the view offset),
// so pass a storage-aligned view to keep npu_stride_copy's storage-relative semantics.
at::Tensor alignToStorageBase(const at::Tensor& self) {
  if (self.storage_offset() == 0) {
    return self;
  }
  return at_npu::native::TransContiguous::view_tensor(self, 0, self.sizes(), self.strides());
}
} // namespace

// acl_op-capable devices (A2/A3) keep the original GE AsStrided implementation;
// aclnn-only devices (Ascend950) materialize the view with aclnnAsStrided.
at::Tensor npu_stride_copy(
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    const at::Scalar& storage_offset) {
  if (!c10_npu::IsAclnnOnly()) {
    return acl_op::npu_stride_copy(self, shape, stride, storage_offset);
  }
  const int64_t storage_offset_value = storage_offset.toLong();
  at::Tensor x = alignToStorageBase(self);
  at::Tensor result = npu_preparation::apply_tensor_without_format(shape, self.options());
  EXEC_NPU_CMD(aclnnAsStrided, x, shape, stride, storage_offset_value, result);
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
  const int64_t storage_offset_value = storage_offset.toLong();
  npu_preparation::check_tensor({self}, out, shape);
  at::Tensor x = alignToStorageBase(self);
  EXEC_NPU_CMD(aclnnAsStrided, x, shape, stride, storage_offset_value, out);
  return out;
}

} // namespace op_api
