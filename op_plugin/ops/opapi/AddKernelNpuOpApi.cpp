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

inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || dtype == at::ScalarType::Bool,
      "Boolean alpha only supported for Boolean results." +
          OPS_ERROR(ErrCode::TYPE));
  TORCH_CHECK(
      isFloatingType(dtype) || isComplexType(dtype) || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating point number." +
          OPS_ERROR(ErrCode::TYPE));
}

static at::Tensor self_tensor_to_device(
    const at::Tensor& tensor,
    const at::ScalarType result_type,
    const c10::Device device) {
  if (npu_preparation::is_scalar_wrapped_to_tensor(tensor) ||
      (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor))) {
    at::Scalar scalar = tensor.item();
    return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
  }
  return tensor;
}

static at::Tensor& add_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  // executing the NPU operator
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar others = other.item();
    EXEC_NPU_CMD(aclnnAdds, self, others, alpha, result);
  } else {
    if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
      // self is a scalar.
      static const bool is_aclnn_available =
          check_aclnn_kernel_available("aclnnAddV3");
      if (is_aclnn_available && self.dtype() != at::kBool && self.dtype() != at::kLong) {
        c10::Scalar selfs = self.item();
        EXEC_NPU_CMD(aclnnAddV3, selfs, other, alpha, result);
      } else {
        at::Tensor self_cp =
            self_tensor_to_device(self, result.scalar_type(), result.device());
        EXEC_NPU_CMD(aclnnAdd, self_cp, other, alpha, result);
      }
    } else {
      // self and other are all npu tensors.
      EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
    }
  }
  return result;
}

static at::Tensor& inplace_add_out_npu_no_check(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // check if other scalar tensor
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    c10::Scalar other_scalar = other.item();
    EXEC_NPU_CMD(aclnnInplaceAdds, self, other_scalar, alpha);
  } else {
    EXEC_NPU_CMD(aclnnInplaceAdd, self, other, alpha);
  }
  return self;
}

static at::Tensor add_dest_output(
    const at::Tensor& self,
    const at::Tensor& other) {
  bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
  return isSelfWrapped ? other : self;
}

// ---------------------------------------------------------------------------
// bool add
//
// For bool operands add is a logical or: True + True is True, while the
// arithmetic kernels write the integer sum 2. Everything bool-specific (alpha
// semantics, operand checks, kernel choice, output allocation) lives in this
// block, and each add overload routes into it up front, so the arithmetic
// path above stays free of bool branches.
// ---------------------------------------------------------------------------

// Bool is the lowest promotion category, so the promoted type is Bool only
// when every operand is Bool. Reading the dtypes keeps the promotion call off
// the non-bool add hot path.
static inline bool add_promotes_to_bool(
    const at::Tensor& self,
    const at::Tensor& other) {
  return self.scalar_type() == at::kBool && other.scalar_type() == at::kBool;
}

static inline bool add_promotes_to_bool(
    const at::Tensor& self,
    const at::Scalar& other) {
  return self.scalar_type() == at::kBool && other.isBoolean();
}

// Bool add is self OR (bool(alpha) AND other), so alpha only decides whether
// other takes part at all.
static inline bool add_bool_alpha_is_zero(const at::Scalar& alpha) {
  alpha_check_npu(at::kBool, alpha);
  return !alpha.to<bool>();
}

// The logical kernels only take tensors, so CPU scalar operands have to be
// put on the device first.
static inline at::Tensor add_bool_operand(
    const at::Tensor& operand,
    const c10::Device device) {
  if (operand.dim() == 0 && !torch_npu::utils::is_npu(operand)) {
    return npu_preparation::copy_scalar_to_device(
        operand.item(), at::kBool, device);
  }
  return operand;
}

// The wording follows the upstream TensorIterator cast check
// (aten/src/ATen/TensorIterator.cpp, enforce_safe_casting_to_output) so a
// rejected in-place bool add reports the same message on npu as on cpu.
template <typename T>
static inline void add_bool_cast_check(const at::Tensor& self, const T& other) {
  const auto result_type = at::native::result_type(self, other);
  TORCH_CHECK(at::canCast(result_type, at::kBool),
      "result type ", result_type,
      " can't be cast to the desired output type Bool", OPS_ERROR(ErrCode::TYPE));
}

static void add_bool_check(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& result) {
  add_bool_cast_check(self, other);
  TORCH_CHECK(
      (self.device() == result.device() || npu_preparation::IsCPUScalar(self)) &&
      (other.device() == result.device() || npu_preparation::IsCPUScalar(other)),
      "Expected all tensors to be on the same device" + OPS_ERROR(ErrCode::PARAM));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  TORCH_CHECK(at::IntArrayRef(output_size) == result.sizes(),
      "output shape does not match the broadcast shape" + OPS_ERROR(ErrCode::PARAM));
}

// The one place a bool add is executed: result aliasing self is the in-place
// form, a non-bool result is the out= form that still has to compute in bool.
static at::Tensor& add_bool_apply(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  if (result.scalar_type() != at::kBool) {
    at::Tensor bool_result = npu_preparation::apply_tensor_without_format(
        result.sizes(), result.options().dtype(at::kBool));
    add_bool_apply(self, other, alpha, bool_result);
    result.copy_(bool_result);
    return result;
  }
  add_bool_check(self, other, result);
  if (add_bool_alpha_is_zero(alpha)) {
    // alpha=0 drops other, but self still goes through the checked broadcast.
    if (!result.is_same(self)) {
      result.copy_(self);
    }
    return result;
  }
  at::Tensor other_b = add_bool_operand(other, result.device());
  if (result.is_same(self)) {
    EXEC_NPU_CMD(aclnnInplaceLogicalOr, result, other_b);
    return result;
  }
  at::Tensor self_b = add_bool_operand(self, result.device());
  EXEC_NPU_CMD(aclnnLogicalOr, self_b, other_b, result);
  return result;
}

static at::Tensor& add_bool_apply(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  // Checked against the raw scalar: a non-bool scalar does not promote to Bool.
  add_bool_cast_check(self, other);
  at::Tensor other_b = npu_preparation::copy_scalar_to_device(
      other, at::kBool, result.device());
  return add_bool_apply(self, other_b, alpha, result);
}

static at::Tensor add_bool(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, add_dest_output(self, other).options().dtype(at::kBool));
  add_bool_apply(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

static at::Tensor add_bool(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto output_size = op_infer::input_same_output_size(self);
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, self.options().dtype(at::kBool));
  add_bool_apply(self, other, alpha, result);
  at::namedinference::propagate_names(result, self);
  return result;
}

static at::Tensor& add_bool_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::check_tensor({self}, result, result, output_size);
  npu_preparation::check_memory({self, other}, {result});
  add_bool_apply(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

static at::Tensor& add_bool_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  npu_preparation::check_memory({self, other}, {self});
  add_bool_apply(self, other, alpha, self);
  at::namedinference::propagate_names_if_nonempty(self, maybe_names);
  return self;
}

static at::Tensor& add_bool_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  add_bool_apply(self, other, alpha, self);
  return self;
}

// The aclop add is arithmetic for bool operands too: it reinterprets them as
// uint8 and runs Add, so True + True lands as 2. The aclnn fallback therefore
// goes to the aclop logical or instead of acl_op::add. logical_or takes no
// alpha; alpha=0 reuses the checked add_bool paths so broadcast, device and
// out= validation still run without depending on aclnnLogicalOr.
static at::Tensor add_bool_fallback(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  add_bool_cast_check(self, other);
  if (add_bool_alpha_is_zero(alpha)) {
    return add_bool(self, other, alpha);
  }
  return acl_op::logical_or(self, other);
}

static at::Tensor add_bool_fallback(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  add_bool_cast_check(self, other);
  if (add_bool_alpha_is_zero(alpha)) {
    return add_bool(self, other, alpha);
  }
  at::Tensor other_b = npu_preparation::copy_scalar_to_device(
      other, at::kBool, self.device());
  return acl_op::logical_or(self, other_b);
}

static at::Tensor& add_bool_fallback_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  add_bool_cast_check(self, other);
  if (add_bool_alpha_is_zero(alpha)) {
    return add_bool_out(self, other, alpha, result);
  }
  if (result.scalar_type() != at::kBool) {
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self}, result, result, output_size);
    at::Tensor bool_result = npu_preparation::apply_tensor_without_format(
        output_size, result.options().dtype(at::kBool));
    add_bool_fallback_out(self, other, alpha, bool_result);
    result.copy_(bool_result);
    return result;
  }
  return acl_op::logical_or_out(self, other, result);
}

static at::Tensor& add_bool_fallback_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  add_bool_cast_check(self, other);
  if (add_bool_alpha_is_zero(alpha)) {
    return add_bool_(self, other, alpha);
  }
  return acl_op::logical_or_(self, other);
}

static at::Tensor& add_bool_fallback_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  add_bool_cast_check(self, other);
  if (add_bool_alpha_is_zero(alpha)) {
    return add_bool_(self, other, alpha);
  }
  at::Tensor other_b = npu_preparation::copy_scalar_to_device(
      other, at::kBool, self.device());
  return acl_op::logical_or_(self, other_b);
}

at::Tensor add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  if (add_promotes_to_bool(self, other)) {
    DO_COMPATIBILITY(aclnnLogicalOr, add_bool_fallback(self, other, alpha));
    return add_bool(self, other, alpha);
  }
  DO_COMPATIBILITY(aclnnAdd, acl_op::add(self, other, alpha));
  DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  // calculate the output size
  at::Tensor output_tensor = add_dest_output(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  alpha_check_npu(result_type, alpha);
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, output_tensor.options().dtype(result_type));
  // calculate the output result of the NPU
  add_out_npu_nocheck(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

at::Tensor add(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (add_promotes_to_bool(self, other)) {
    DO_COMPATIBILITY(aclnnLogicalOr, add_bool_fallback(self, other, alpha));
    return add_bool(self, other, alpha);
  }
  DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
  // calculate the output size
  auto output_size = op_infer::input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  alpha_check_npu(result_type, alpha);
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(
      output_size, self.options().dtype(result_type));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnAdds, self, other, alpha, result);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& result) {
  if (add_promotes_to_bool(self, other) || result.scalar_type() == at::kBool) {
    DO_COMPATIBILITY(aclnnLogicalOr, add_bool_fallback_out(self, other, alpha, result));
    return add_bool_out(self, other, alpha, result);
  }
  DO_COMPATIBILITY(aclnnAdd, acl_op::add_out(self, other, alpha, result));
  DO_COMPATIBILITY(aclnnAdds, acl_op::add_out(self, other, alpha, result));
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
  at::Tensor output_tensor = isSelfWrapped ? other : self;
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);

  npu_preparation::check_tensor({self}, result, result, output_size);
  npu_preparation::check_memory({self, other}, {result});
  add_out_npu_nocheck(self, other, alpha, result);
  at::namedinference::propagate_names_if_nonempty(result, maybe_names);
  return result;
}

at::Tensor& add_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  if (self.scalar_type() == at::kBool) {
    DO_COMPATIBILITY(aclnnInplaceLogicalOr, add_bool_fallback_(self, other, alpha));
    return add_bool_(self, other, alpha);
  }
  DO_COMPATIBILITY(aclnnInplaceAdd, acl_op::add_(self, other, alpha));
  DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
  std::vector<at::Tensor> tensor_list = {self, other};
  auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
  npu_preparation::check_memory({self, other}, {self});
  inplace_add_out_npu_no_check(self, other, alpha);
  at::namedinference::propagate_names_if_nonempty(self, maybe_names);
  return self;
}

at::Tensor& add_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (self.scalar_type() == at::kBool) {
    DO_COMPATIBILITY(aclnnInplaceLogicalOr, add_bool_fallback_(self, other, alpha));
    return add_bool_(self, other, alpha);
  }
  DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
  EXEC_NPU_CMD(aclnnInplaceAdds, self, other, alpha);
  return self;
}
} // namespace op_api
