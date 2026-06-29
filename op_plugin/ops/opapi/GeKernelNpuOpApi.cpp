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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGeScalar, acl_op::ge_out(self, other, result));
  auto output_size = self.sizes();
  npu_preparation::check_tensor({self}, result, output_size);
  EXEC_NPU_CMD(aclnnGeScalar, self, other, result);
  return result;
}

at::Tensor ge(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnGeScalar, acl_op::ge(self, other));
  auto output_size = op_infer::input_same_output_size(self);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnGeScalar, self, other, result);
  return result;
}

at::Tensor& ge_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceGeScalar, acl_op::ge_(self, other));
  EXEC_NPU_CMD(aclnnInplaceGeScalar, self, other);
  return self;
}

at::Tensor& ge_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGeTensor, acl_op::ge_out(self, other, result));
  if (is_ascend950_path()) {
      auto [self_device, other_device] = prepare_binary_tensors(self, other);
      auto maybe_names = op_plugin::utils::compute_names_npu({self, other});
      auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
      npu_preparation::check_tensor({self_device, other_device}, result, output_size);
      EXEC_NPU_CMD(aclnnGeTensor, self_device, other_device, result);
      at::namedinference::propagate_names_if_nonempty(result, maybe_names);
      return result;
  }
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self, other}, result, output_size);
  EXEC_NPU_CMD(aclnnGeTensor, self, other, result);
  return result;
}

at::Tensor ge(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnGeTensor, acl_op::ge(self, other));
  if (is_ascend950_path()) {
      auto [self_device, other_device] = prepare_binary_tensors(self, other);
      auto maybe_names = op_plugin::utils::compute_names_npu({self, other});
      auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
      at::Tensor result = npu_preparation::apply_tensor_without_format(
          output_size, self_device.options().dtype(at::kBool));
      EXEC_NPU_CMD(aclnnGeTensor, self_device, other_device, result);
      at::namedinference::propagate_names_if_nonempty(result, maybe_names);
      return result;
  }
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    DO_COMPATIBILITY(aclnnGeScalar, acl_op::ge(self, other));
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kBool));
    const at::Scalar tmpItem = other.item();
    EXEC_NPU_CMD(aclnnGeScalar, self, tmpItem, result);
    return result;
  } else if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
    DO_COMPATIBILITY(aclnnLessScalar, acl_op::ge(self, other));
    at::Tensor result = npu_preparation::apply_tensor_without_format(other.sizes(), other.options().dtype(at::kBool));
    const at::Scalar tmpItem = self.item();
    EXEC_NPU_CMD(aclnnLessScalar, other, tmpItem, result);
    return result;
  } else {
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnGeTensor, self, other, result);
    return result;
  }
}

at::Tensor& ge_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceGeTensor, acl_op::ge_(self, other));
    if (is_ascend950_path()) {
        TORCH_CHECK(torch_npu::utils::is_npu(self),
            "inplace ge_ requires self to be NPU tensor", OPS_ERROR(ErrCode::PARAM));
        at::Tensor other_device = other;
        if (!torch_npu::utils::is_npu(other)) {
            other_device = other.to(self.device());
        }
        npu_preparation::CheckMemory({self, other_device}, {self});
        EXEC_NPU_CMD(aclnnInplaceGeTensor, self, other_device);
        return self;
    }
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        return op_api::ge_(self, other.item());
    } else {
        TORCH_CHECK(self.device() == other.device(),
                    "Expected all tensors to be on the same device, but found at least two devices", OPS_ERROR(ErrCode::INTERNAL));
        npu_preparation::CheckMemory({self, other}, {self});
        EXEC_NPU_CMD(aclnnInplaceGeTensor, self, other);
        return self;
    }
}
}
