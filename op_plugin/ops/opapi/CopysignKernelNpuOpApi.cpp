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

#include <limits>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::ScalarType copysign_result_type(const at::Tensor& self, const at::Tensor& other) {
  at::ScalarType high_type;

  if (npu_preparation::IsCPUScalar(other)) {
    high_type = at::native::result_type(self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    high_type = at::native::result_type(other, self.item());
  } else {
    high_type = at::native::result_type(self, other);
  }
  // 对齐 TensorIterator 的 promote_integer_inputs_to_float 逻辑：整型（包括 bool）
  // 输入会被提升为默认浮点 dtype（遵循 torch.set_default_dtype）。
  if (isIntegralType(high_type, true)) {
    return c10::get_default_dtype_as_scalartype();
  }
  return high_type;
}

void copysign_check_out_dtype(at::ScalarType high_type, at::ScalarType out_type) {
  TORCH_CHECK(
      !(isFloatingType(high_type) && isIntegralType(out_type, true)),
      "result type ",
      high_type,
      " can't be cast to the desired output type ",
      out_type,
      OPS_ERROR(ErrCode::TYPE));
}

// 用整数运算实现位精确的 IEEE-754 copysign：
//     result_bits = (self_bits & 0x7FFF...) | (other_bits & 0x8000...)
at::ScalarType copysign_bits_type(at::ScalarType float_type) {
  switch (float_type) {
    case at::kDouble:
      return at::kLong;
    case at::kFloat:
      return at::kInt;
    case at::kHalf:
    case at::kBFloat16:
      return at::kShort;
    default:
      TORCH_CHECK(false, "copysign: unsupported compute dtype ", float_type, OPS_ERROR(ErrCode::TYPE));
      return at::kLong;
  }
}

void copysign_npu_impl(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  const auto bits_type = copysign_bits_type(self.scalar_type());
  const bool is64 = bits_type == at::kLong;
  const bool is32 = bits_type == at::kInt;
  const int64_t abs_mask = is64 ? std::numeric_limits<int64_t>::max() : (is32 ? 0x7FFFFFFFLL : 0x7FFFLL);
  const int64_t sign_mask = is64 ? std::numeric_limits<int64_t>::min()
                                 : (is32 ? static_cast<int64_t>(std::numeric_limits<int32_t>::min())
                                         : static_cast<int64_t>(std::numeric_limits<int16_t>::min()));
  auto self_bits = self.view(bits_type);
  auto other_bits = other.view(bits_type);
  auto abs_bits = self_bits.bitwise_and(abs_mask);
  auto sign_bits = other_bits.bitwise_and(sign_mask);
  auto out_bits = out.view(bits_type);
  at::bitwise_or_out(out_bits, abs_bits, sign_bits);
}
} // namespace

at::Tensor& copysign_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  auto high_type = copysign_result_type(self, other);
  // NPU 无法将整型输入转换为 double（To 算子会把 double 静默转换为 float，
  // 这会破坏下面位精确的计算并返回错误结果）。double 结果只有在两个输入都
  // 已经是浮点 double 张量时才会出现；因此拒绝整型 -> double 的路径。
  TORCH_CHECK(
      high_type != at::kDouble ||
          (!isIntegralType(self.scalar_type(), true) &&
           !isIntegralType(other.scalar_type(), true)),
      "copysign: double dtype is not supported on NPU",
      OPS_ERROR(ErrCode::TYPE));
  copysign_check_out_dtype(high_type, out.scalar_type());
  auto [self_device, other_device] = prepare_binary_tensors(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self_device, other_device);
  npu_preparation::check_tensor({self_device, other_device}, out, out.scalar_type(), output_size);

  if (out.scalar_type() == high_type) {
    copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), out);
  } else {
    at::Tensor tmp = npu_preparation::apply_tensor_without_format(
        output_size, self_device.options().dtype(high_type));
    copysign_npu_impl(self_device.to(high_type), other_device.to(high_type), tmp);
    out.copy_(tmp);
  }
  return out;
}
} // namespace op_api
