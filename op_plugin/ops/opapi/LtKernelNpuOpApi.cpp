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
at::Tensor& lt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLtTensor, acl_op::lt_out(self, other, result));
  if (is_ascend950_path()) {
      auto [self_device, other_device] = prepare_binary_tensors(self, other);
      auto maybe_names = op_plugin::utils::compute_names_npu({self, other});
      auto outputSize = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
      npu_preparation::check_tensor({self_device, other_device}, result, outputSize);
      EXEC_NPU_CMD(aclnnLtTensor, self_device, other_device, result);
      at::namedinference::propagate_names_if_nonempty(result, maybe_names);
      return result;
  }
  auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);

  at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);

  if (npu_preparation::IsCPUScalar(self)) {
      const at::Scalar self_scalar = self.item();
      EXEC_NPU_CMD(aclnnGtScalar, other, self_scalar, result);
  } else if (npu_preparation::IsCPUScalar(other)) {
      const at::Scalar other_scalar = other.item();
      EXEC_NPU_CMD(aclnnLtScalar, self, other_scalar, result);
  } else {
      EXEC_NPU_CMD(aclnnLtTensor, self, other, result);
  }
  return result;
}

at::Tensor lt(const at::Tensor& self, const at::Tensor& other) {
    DO_COMPATIBILITY(aclnnLtTensor, acl_op::lt(self, other));
    if (is_ascend950_path()) {
        auto [self_device, other_device] = prepare_binary_tensors(self, other);
        auto maybe_names = op_plugin::utils::compute_names_npu({self, other});
        auto outputSize = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
        at::Tensor result = npu_preparation::apply_tensor_without_format(
            outputSize, self_device.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnLtTensor, self_device, other_device, result);
        at::namedinference::propagate_names_if_nonempty(result, maybe_names);
        return result;
    }
    // calculate the output size
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);

    // construct the output tensor of the NPU
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(outputSize,
                                                                                   self.options().dtype(at::kBool));

    // calculate the output result of the NPU
    if (npu_preparation::IsCPUScalar(self)) {
        const at::Scalar self_scalar = self.item();
        EXEC_NPU_CMD(aclnnGtScalar, other, self_scalar, result);
    } else if (npu_preparation::IsCPUScalar(other)) {
        const at::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnLtScalar, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnLtTensor, self, other, result);
    }
    return result;
}

at::Tensor& lt_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnLtScalar, acl_op::lt_out(self, other, result));
  auto outputSize = self.sizes();
  at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);

  EXEC_NPU_CMD(aclnnLtScalar, self, other, result);
  return result;
}

at::Tensor lt(const at::Tensor &self, const at::Scalar& other)
{
  DO_COMPATIBILITY(aclnnLtScalar, acl_op::lt(self, other));
  // calculate the output size
  auto outputSize = op_infer::input_same_output_size(self);
  // construct the output tensor of the NPU
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(outputSize,
                                                                                 self.options().dtype(at::kBool));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLtScalar, self, other, result);
  return result;
}

at::Tensor& lt_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceLtTensor, acl_op::lt_(self, other));
  if (is_ascend950_path()) {
      TORCH_CHECK(torch_npu::utils::is_npu(self),
          "inplace lt_ requires self to be NPU tensor", OPS_ERROR(ErrCode::PARAM));
      at::Tensor other_device = other;
      if (!torch_npu::utils::is_npu(other)) {
          other_device = other.to(self.device());
      }
      npu_preparation::CheckMemory({self, other_device}, {self});
      EXEC_NPU_CMD(aclnnInplaceLtTensor, self, other_device);
      return self;
  }
  if (npu_preparation::IsCPUScalar(other)) {
      return op_api::lt_(self, other.item());
  } else {
      EXEC_NPU_CMD(aclnnInplaceLtTensor, self, other);
      return self;
  }
}

at::Tensor& lt_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceLtScalar, acl_op::lt_(self, other));
  EXEC_NPU_CMD(aclnnInplaceLtScalar, self, other);
  return self;
}

}
