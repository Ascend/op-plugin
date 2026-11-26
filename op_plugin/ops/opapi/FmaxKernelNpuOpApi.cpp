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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

using npu_preparation = at_npu::native::OpPreparation;

namespace op_api {

// CANN 无 aclnnFmax，浮点按上游 decomposition 组合实现，整数/bool 等价 maximum。
inline bool _fmax_combo_symbols_available() {
  static const bool available = check_aclnn_kernel_available("aclnnNeTensor") &&
      check_aclnn_kernel_available("aclnnLtTensor") &&
      check_aclnn_kernel_available("aclnnLogicalOr") &&
      check_aclnn_kernel_available("aclnnSWhere") &&
      check_aclnn_kernel_available("aclnnMaximum");
  return available;
}

// complex 需在设备搬运之前报错，文案与上游 TORCH_META_FUNC(fmax) 一致。
inline void _fmax_check_complex(const at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK_TYPE(
      !self.is_complex() && !other.is_complex(),
      "fmax not implemented for complex tensors.");
}

inline bool _fmax_fallback_condition(const at::Tensor& self, const at::Tensor& other) {
  static const bool is_support_soc =
      (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
      (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
  if (!is_support_soc || !_fmax_combo_symbols_available()) {
    return true;
  }
  // double 走组合接口虽可算，但大 shape 下慢于 fallback，本期不做原生。
  const at::ScalarType high_type = at::native::result_type(self, other);
  if (high_type == at::ScalarType::Double) {
    return true;
  }
  return false;
}

// mask 形状可广播到 output_size，result 允许与输入共享内存。
static at::Tensor& fmax_combo_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result,
    const at::IntArrayRef& output_size) {
  at::Tensor nan_mask = npu_preparation::apply_tensor_without_format(
      other.sizes(), other.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnNeTensor, other, other, nan_mask);  // isnan(other)
  at::Tensor lt_mask = npu_preparation::apply_tensor_without_format(
      output_size, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLtTensor, other, self, lt_mask);  // other < self
  at::Tensor mask = npu_preparation::apply_tensor_without_format(
      output_size, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLogicalOr, nan_mask, lt_mask, mask);
  EXEC_NPU_CMD(aclnnSWhere, mask, self, other, result);
  return result;
}

inline at::Tensor& _fmax_launch(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result,
    at::ScalarType result_dtype,
    const at::IntArrayRef& output_size) {
  if (c10::isFloatingType(result_dtype)) {
    return fmax_combo_out(self, other, result, output_size);
  }
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

at::Tensor& fmax_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  _fmax_check_complex(self, other);
  auto [self_device, other_device] = prepare_binary_tensors(self, other);
  at::ScalarType result_dtype = at::native::result_type(self_device, other_device);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
  // 需在 fallback 之前校验：op_plugin 直接注册实现，绕过了上游 meta 的 resize。
  npu_preparation::check_tensor({self_device, other_device}, result, output_size);
  if (_fmax_fallback_condition(self, other)) {
    // 用原始输入计算，避免 CPU 标量被搬到 device 后再搬回。
    at::Tensor self_cpu = self.cpu();
    at::Tensor other_cpu = other.cpu();
    at::Tensor result_cpu = result.cpu();
    at::fmax_outf(self_cpu, other_cpu, result_cpu);
    result.copy_(result_cpu);
    return result;
  }
  if (result.scalar_type() != result_dtype) {
    TORCH_CHECK(
        canCast(result_dtype, result.scalar_type()),
        "result type ",
        result_dtype,
        " can't be cast to the desired output type ",
        result.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    at::Tensor tmp = npu_preparation::apply_tensor_without_format(
        output_size, self_device.options().dtype(result_dtype));
    _fmax_launch(self_device, other_device, tmp, result_dtype, output_size);
    result.copy_(tmp);
    return result;
  }
  return _fmax_launch(self_device, other_device, result, result_dtype, output_size);
}

} // namespace op_api
