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
static const int MODE_TRUNC = 1;
static const int MODE_FLOOR = 2;

static void check_rounding_mode_npu(c10::optional<c10::string_view> rounding_mode)
{
    TORCH_CHECK((!rounding_mode.has_value() || *rounding_mode == "trunc" || *rounding_mode == "floor"),
                "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
                "but found '",
                *rounding_mode, "'", OPS_ERROR(ErrCode::PARAM));
}

static at::Tensor& div_out_npu_opapi_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  // executing the NPU operator
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar others = other.item();
    EXEC_NPU_CMD(aclnnDivs, self, others, result);
  } else {
    EXEC_NPU_CMD(aclnnDiv, self, other, result);
  }
  return result;
}

static at::Tensor self_tensor_to_device(const at::Tensor& tensor, const at::ScalarType result_type,
                                        const c10::Device device)
{
  if (npu_preparation::is_scalar_wrapped_to_tensor(tensor)) {
    at::Scalar scalar = tensor.item();
    return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
  }
  return tensor;
}

at::Tensor& div_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnDivs, acl_op::div_out(self, other, result));
    DO_COMPATIBILITY(aclnnDiv, acl_op::div_out(self, other, result));
    // calculate the output size
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    if (!isFloatingType(result_type) && !isComplexType(result_type)) {
        result_type = at::ScalarType::Float;
    }
    if (isFloatingType(result.scalar_type()) || isComplexType(result.scalar_type())) {
        result_type = result.scalar_type();
    }
    at::Tensor self_cp = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self}, result, result_type, output_size);

    // calculate the output result of the NPU
    div_out_npu_opapi_nocheck(self_cp, other, result);
    return result;
}

at::Tensor& div_out(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode,
                    at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnDivMods, acl_op::div_out(self, other, rounding_mode, result));
    DO_COMPATIBILITY(aclnnDivMod, acl_op::div_out(self, other, rounding_mode, result));
    if (rounding_mode.has_value() && *rounding_mode != "floor" && *rounding_mode != "trunc") {
        TORCH_CHECK(false,
                    "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
                    "but found '",
                    *rounding_mode, "'", OPS_ERROR(ErrCode::PARAM));
    }

    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self}, result, result.scalar_type(), outputSize);

    int mode = 0;
    if (rounding_mode.has_value() && *rounding_mode == "floor") {
        mode = MODE_FLOOR;
    } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
        mode = MODE_TRUNC;
    }

    // calculate the output result of the NPU
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnDivMods, self_cp, others, mode, result);
    } else {
        EXEC_NPU_CMD(aclnnDivMod, self_cp, other, mode, result);
    }
    return result;
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnDivs, acl_op::div(self, other));
  DO_COMPATIBILITY(aclnnDiv, acl_op::div(self, other));
  // calculate the output size
  bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
  at::Tensor outputTensor = isSelfWrapped ? other : self;
  auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_cp = self_tensor_to_device(self, high_type, outputTensor.device());

  if (!isFloatingType(high_type) && !isComplexType(high_type)) {
    high_type = at::ScalarType::Float;
  }
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, outputTensor.options().dtype(high_type));

  // calculate the output result of the NPU
  div_out_npu_opapi_nocheck(self_cp, other, result);
  return result;
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode)
{
    DO_COMPATIBILITY(aclnnDivMods, acl_op::div(self, other, rounding_mode));
    DO_COMPATIBILITY(aclnnDivMod, acl_op::div(self, other, rounding_mode));
    if (rounding_mode.has_value() && *rounding_mode != "floor" && *rounding_mode != "trunc") {
        TORCH_CHECK(false,
                    "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
                    "but found '",
                    *rounding_mode, "'", OPS_ERROR(ErrCode::PARAM));
    }

    // calculate the output size
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor outputTensor = isSelfWrapped ? other : self;

    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, high_type, outputTensor.device());

    // construct the output tensor of the NPU
    int mode = 0;
    if (rounding_mode.has_value() && *rounding_mode == "floor") {
        mode = MODE_FLOOR;
    } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
        mode = MODE_TRUNC;
    } else {
        if (!isFloatingType(high_type) && !isComplexType(high_type)) {
            high_type = at::ScalarType::Float;
        }
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, outputTensor.options().dtype(high_type));

    // executing the NPU operator
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnDivMods, self_cp, others, mode, result);
    } else {
        EXEC_NPU_CMD(aclnnDivMod, self_cp, other, mode, result);
    }
    return result;
}

static at::Tensor& inplace_div_out_npu_no_check(at::Tensor& self, const at::Tensor& other) {
  // check if other scalar tensor
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar others = other.item();
    EXEC_NPU_CMD(aclnnInplaceDivs, self, others);
  } else {
    EXEC_NPU_CMD(aclnnInplaceDiv, self, other);
  }
  return self;
}

static at::Tensor& inplace_div_out_mode_npu_no_check(at::Tensor& self, const at::Tensor& other, int mode) {
  // check if other scalar tensor
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar others = other.item();
    EXEC_NPU_CMD(aclnnInplaceDivMods, self, others, mode);
  } else {
    EXEC_NPU_CMD(aclnnInplaceDivMod, self, other, mode);
  }
  return self;
}

at::Tensor& div_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceDivs, acl_op::div_(self, other));
  DO_COMPATIBILITY(aclnnInplaceDiv, acl_op::div_(self, other));
  const std::initializer_list<at::Tensor> inputs = {self, other};
  const std::initializer_list<at::Tensor> outputs = {self};
  npu_preparation::check_memory(inputs, outputs);
  inplace_div_out_npu_no_check(self, other);
  return self;
}

at::Tensor& div_(at::Tensor& self, const at::Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnInplaceDivMods, acl_op::div_(self, other, rounding_mode));
  DO_COMPATIBILITY(aclnnInplaceDivMod, acl_op::div_(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  const std::initializer_list<at::Tensor> inputs = {self, other};
  const std::initializer_list<at::Tensor> outputs = {self};
  npu_preparation::check_memory(inputs, outputs);
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = MODE_FLOOR;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = MODE_TRUNC;
  }
  inplace_div_out_mode_npu_no_check(self, other, mode);
  return self;
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnDivs, acl_op::div(self, other));
  auto outputSize = op_infer::input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  if (!isFloatingType(high_type) && !isComplexType(high_type)) {
    high_type = at::ScalarType::Float;
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnDivs, self, other, result);
  return result;
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnDivMods, acl_op::div(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  auto outputSize = op_infer::input_same_output_size(self);
  at::ScalarType high_type = at::native::result_type(self, other);
  // construct the output tensor of the NPU
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = MODE_FLOOR;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = MODE_TRUNC;
  } else {
    if (!isFloatingType(high_type) && !isComplexType(high_type)) {
      high_type = at::ScalarType::Float;
    }
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(high_type));
  EXEC_NPU_CMD(aclnnDivMods, self, other, mode, result);
  return result;
}

at::Tensor& div_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceDivs, acl_op::div_(self, other));
  EXEC_NPU_CMD(aclnnInplaceDivs, self, other);
  return self;
}

at::Tensor& div_(at::Tensor& self, const at::Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  DO_COMPATIBILITY(aclnnInplaceDivMods, acl_op::div_(self, other, rounding_mode));
  check_rounding_mode_npu(rounding_mode);
  int mode = 0;
  if (rounding_mode.has_value() && *rounding_mode == "floor") {
    mode = MODE_FLOOR;
  } else if (rounding_mode.has_value() && *rounding_mode == "trunc") {
    mode = MODE_TRUNC;
  }
  EXEC_NPU_CMD(aclnnInplaceDivMods, self, other, mode);
  return self;
}

}
