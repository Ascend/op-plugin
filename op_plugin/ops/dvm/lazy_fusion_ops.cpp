// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/record_function.h>
#include "op_plugin/DvmOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/ops/dvm/lazy_fusion_kernel.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace lazy_fusion {
namespace {
using ShapeVector = std::vector<int64_t>;

// aclnn rounds the scalar argument of add_scalar/sub_scalar to the input dtype
// (bf16/fp16) before the op; for mul/div the scalar stays fp32. Mirror that
// op-by-op so DVM ON matches DVM OFF (aclnn) bit-exactly on each path.
inline float RoundScalarToInputDtype(float scalar, at::ScalarType self_dtype) {
  if (self_dtype == at::ScalarType::BFloat16) {
    return static_cast<float>(static_cast<at::BFloat16>(scalar));
  }
  if (self_dtype == at::ScalarType::Half) {
    return static_cast<float>(static_cast<at::Half>(scalar));
  }
  return scalar;
}

bool IsFloat16_32(const at::ScalarType &type) {
  return type == at::ScalarType::Half || type == at::ScalarType::Float;
}

bool IsFloatType(const at::ScalarType &type) {
  return type == at::ScalarType::Half || type == at::ScalarType::Float || type == at::ScalarType::BFloat16;
}

bool IsFloatIntType(const at::ScalarType &type) { return IsFloatType(type) || type == at::ScalarType::Int; }

bool IsSupportType(const at::ScalarType &type) { return IsFloatIntType(type) || type == at::ScalarType::Bool; }

// DVM's strided Load (ViewLoad) is implemented for sub-rectangular blocks of a
// contiguous storage where the innermost load-carrying dim has unit stride.
// Aligned with MindSpore's IsViewLoadSupported (same DVM library underneath).
//
// Key correctness gate: `sizes[-1] != 1`. Without it, views with a trailing
// unsqueezed dim (size==1, stride==1) would slip past a naive `strides[-1]==1`
// check while the *real* load-carrying dim has stride>1. Concrete GLM4.7 RoPE
// example:
//   shape  = [8192, 1, 16, 32, 1]
//   stride = [1024, 1024, 64,  2, 1]
// strides[-1]==1 only because sizes[-1]==1. The actual load-carrying dim is
// shape[3]=32 with stride=2 (an `x.view(..., D/2, 2)[..., 0]` even/odd slice).
// DVM cannot ViewLoad with stride=2 and silently corrupts the matmul output,
// which manifested as GLM4.7 forward loss going wrong.
//
// Negative-stride and zero/negative-size views are rejected as well because
// DVM's address arithmetic assumes non-negative strides over real elements.
bool IsViewLoadable(const at::Tensor &x) {
  if (!x.defined()) {
    return false;
  }
  auto ndim = x.dim();
  if (ndim == 0) {
    return false;
  }
  auto sizes = x.sizes();
  auto strides = x.strides();
  for (int64_t i = 0; i < ndim; ++i) {
    if (sizes[i] <= 0 || strides[i] < 0) {
      return false;
    }
  }
  return strides[ndim - 1] == 1 && sizes[ndim - 1] != 1;
}

bool InputCheck(const at::Tensor &x, const std::function<bool(const at::ScalarType &type)> &type_check = IsFloatType,
                bool allow_non_contig = false) {
  if (!x.is_contiguous()) {
    // ViewLoad on non-contig inputs is an O2-level optimization
    // (same gating as the historical enable_all switch).
    bool can_viewload = g_lazy_fusion_manager.flags_.level >= Level::kO2 &&
                        allow_non_contig && IsViewLoadable(x);
    if (!can_viewload) {
      return false;
    }
  }
  return torch_npu::utils::is_npu(x) && type_check(x.scalar_type());
}

bool IsCPUScalar(const at::Tensor &x) {
  return x.defined() && x.dim() == 0 && x.device().type() == c10::DeviceType::CPU;
}

// Every tensor input of every DVM op must go through an NPU device guard before
// entering the lazy-fusion path. Use NpuCheck for pure device checks, or use
// InputCheck when the op also needs the contiguous/type constraints that it
// enforces. This includes optional tensor inputs when they are present. Do not
// open-code partial device checks at call sites, otherwise a non-NPU tensor can
// slip into k->Input/k->Output and cause an invalid execution path.
bool NpuCheck(const at::Tensor &x) {
  return x.defined() && torch_npu::utils::is_npu(x);
}

bool NpuCheck(const c10::optional<at::Tensor> &x) {
  return !x.has_value() || NpuCheck(x.value());
}

template <typename T, typename... Args>
bool NpuCheck(const T &x, const Args &...xs) {
  return NpuCheck(x) && NpuCheck(xs...);
}

bool GetScalarValue(const at::Scalar &scalar, float *v) {
  auto scalar_type = scalar.type();
  if (scalar_type == at::ScalarType::Long) {
    *v = static_cast<float>(scalar.toLong());
  } else if (scalar_type == at::ScalarType::Double) {
    *v = static_cast<float>(scalar.toDouble());
  } else {
    return false;
  }
  return true;
}

constexpr float kTanhClampMin = -8.8f;
constexpr float kTanhClampMax = 8.8f;

dvm::NDObject *BuildClampedTanhFp32(dvm::Kernel *k, dvm::NDObject *input_f32) {
  auto clamped = k->Binary<dvm::BinaryType::kMaximum>(input_f32, kTanhClampMin);
  clamped = k->Binary<dvm::BinaryType::kMinimum>(clamped, kTanhClampMax);
  auto two_x = k->Binary<dvm::BinaryType::kMul>(clamped, 2.0f);
  auto exp_two_x = k->Unary<dvm::UnaryType::kExp>(two_x);
  auto numer = k->Binary<dvm::BinaryType::kSub>(exp_two_x, 1.0f);
  auto denom = k->Binary<dvm::BinaryType::kAdd>(exp_two_x, 1.0f);
  return k->Binary<dvm::BinaryType::kDiv>(numer, denom);
}

// Call this for every tensor that the current fused op reads through k->Input(...),
// unless the op entry already does an unconditional LazyFusionFlush() before any
// graph mutation. The helper checks whether the cached lazy_fusion_data_ still represents
// the exact same tensor metadata + logical DVM shape, or whether the old value
// was only a fusion value that can no longer be rebuilt safely from GM.
//
// Typical cases that should call PrepareFusionInput:
//   - normal read-only inputs of unary/binary/reduce ops
//   - tensors reused across fused ops with a different logical DVM shape
//   - tensors that may hide an unseen view/as_strided/set_ between two captured ops
//
// Typical cases that do not need it:
//   - entries like MatMul / batch_norm_* that unconditionally flush first and
//     then start a brand new fusion graph
//
// Example:
//   x0 = torch.linspace(-400, 399, steps=800, dtype=torch.float16).reshape(4, 200).npu()
//   y0 = abs(x0)
//   y1 = y0[0, :]
//   y2 = add(y0, y1)  // need flush before this op
//
// PTA only sees abs/add here. When add asks for y1, the cached lazy_fusion_data_ points
// to y0's fusion value, but the slice view between them is invisible in this repo
// and cannot be rebuilt from GM. The flush must happen before add starts building,
// otherwise Input(y1) would hit a non-reloadable alias miss after the current op
// has already mutated the DVM graph.
void PrepareFusionInput(const at::Tensor &tensor) {
  if (tensor.defined() && torch_npu::utils::is_npu(tensor) && g_lazy_fusion_manager.NeedFlushForInput(tensor)) {
    LazyFusionFlush();
  }
}

void PrepareFusionInput(const c10::optional<at::Tensor> &tensor) {
  if (tensor.has_value()) {
    PrepareFusionInput(tensor.value());
  }
}

void PrepareFusionInput(at::TensorList tensors) {
  for (const auto &tensor : tensors) {
    PrepareFusionInput(tensor);
  }
}

template <typename... Args>
void PrepareFusionInputs(const Args &... inputs) {
  (PrepareFusionInput(inputs), ...);
}

// Call this for every tensor that the current op may overwrite in place through
// k->Output(tensor, ..., true) or an out= style writable output. Writable tensors
// are stricter than read-only inputs: anything other than the exact same tensor
// metadata is conservatively flushed before the current op is added to the graph.
//
// Typical cases that should call PrepareWritableOutput:
//   - inplace ops such as add_, mul_, exp_, _foreach_sqrt_
//   - out= variants that write into a user-provided tensor
//
// Typical cases that do not need it:
//   - ops that return a freshly allocated tensor through k->Output(obj, shape, options)
//   - entries that already flush unconditionally before they start building a graph
//
// Example:
//   y0 = mul(x0, x1)
//   y1 = y0[100:200]
//   y2 = inplace_add(y0, x2)  // need flush before this op
//
// The write target y0 shares storage with another in-flight fusion value through
// a non-exact alias chain. If inplace_add is built without flushing, the old store
// and the new inplace store may overlap in GM and trigger the precision issue seen
// in DVM. The flush must happen before the inplace op is emitted, not later in
// Output(), because by then the current op has already been appended to the graph.
void PrepareWritableOutput(const at::Tensor &tensor) {
  if (tensor.defined() && torch_npu::utils::is_npu(tensor) && g_lazy_fusion_manager.NeedFlushForWritableOutput(tensor)) {
    LazyFusionFlush();
  }
}

void PrepareWritableOutput(at::TensorList tensors) {
  for (const auto &tensor : tensors) {
    PrepareWritableOutput(tensor);
  }
}

bool AddScalar(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha, at::Tensor *out);

bool Add(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor *out = nullptr) {
  if (IsCPUScalar(other)) {
    return AddScalar(self, other.item(), alpha, out);
  }
  bool inplace = (out == nullptr);
  if (!InputCheck(self, IsFloatIntType, /*allow_non_contig=*/!inplace) ||
      !InputCheck(other, IsFloatIntType, /*allow_non_contig=*/true)) {
    return false;
  }
  float scalar = 1.0f;
  auto alpha_type = alpha.type();
  if (alpha_type == at::ScalarType::Long) {
    scalar = static_cast<float>(alpha.toLong());
  } else if (alpha_type == at::ScalarType::Double) {
    scalar = static_cast<float>(alpha.toDouble());
  } else {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
    at_npu::native::OpPreparation::check_memory({self, other}, {self});
  }
  PrepareFusionInputs(self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto result_type = at::native::result_type(self, other);
  auto compute_type = result_type == at::ScalarType::BFloat16 ? dvm::DType::kFloat32 : k->TransType(result_type);
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  self_obj = k->Cast(self_obj, compute_type);
  other_obj = k->Cast(other_obj, compute_type);
  if (scalar != 1.0f) {
    other_obj = k->Binary<dvm::BinaryType::kMul>(other_obj, scalar);
  }
  auto out_obj = k->Binary<dvm::BinaryType::kAdd>(self_obj, other_obj);
  if (out == nullptr) {
    k->Output(self, out_obj, true);
  } else {
    *out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(result_type));
  }
  return true;
}

bool Sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor *out = nullptr) {
  bool inplace = (out == nullptr);
  if (!InputCheck(self, IsFloatIntType, /*allow_non_contig=*/!inplace) ||
      !InputCheck(other, IsFloatIntType, /*allow_non_contig=*/true)) {
    return false;
  }
  float scalar = 1.0f;
  bool cast_fp32 = false;
  auto alpha_type = alpha.type();
  if (alpha_type == at::ScalarType::Long) {
    scalar = static_cast<float>(alpha.toLong());
  } else if (alpha_type == at::ScalarType::Double) {
    scalar = static_cast<float>(alpha.toDouble());
    cast_fp32 = true;
  } else {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
    at_npu::native::OpPreparation::check_memory({self, other}, {self});
  }
  PrepareFusionInputs(self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto result_type = at::native::result_type(self, other);
  auto compute_type = (result_type == at::ScalarType::BFloat16 || cast_fp32) ? dvm::DType::kFloat32 : k->TransType(result_type);
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  self_obj = k->Cast(self_obj, compute_type);
  other_obj = k->Cast(other_obj, compute_type);
  if (scalar != 1.0f) {
    other_obj = k->Binary<dvm::BinaryType::kMul>(other_obj, scalar);
  }
  auto out_obj = k->Binary<dvm::BinaryType::kSub>(self_obj, other_obj);
  if (out == nullptr) {
    k->Output(self, out_obj, true);
  } else {
    *out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(result_type));
  }
  return true;
}

bool BinaryScalar(const at::Tensor &self, const at::Scalar &other, dvm::BinaryOpType op_type,
            const std::function<bool(const at::ScalarType &type)> &type_check,
            dvm::DType dst_type = dvm::DType::kDataTypeEnd, at::Tensor *out = nullptr) {
  // out == nullptr is the inplace path (writes back to self) → must be contig.
  // out != nullptr is the functional path (fresh output) → self is read-only,
  // can be a non-contig view; Input() will route to a strided Load.
  bool allow_non_contig = (out != nullptr);
  if (!InputCheck(self, type_check, allow_non_contig)) {
    return false;
  }
  float scalar = 1.0f;
  auto other_type = other.type();
  if (other_type == at::ScalarType::Long) {
    scalar = static_cast<float>(other.toLong());
  } else if (other_type == at::ScalarType::Double) {
    scalar = static_cast<float>(other.toDouble());
  } else {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto result_type = at::native::result_type(self, other);
  auto compute_type = result_type == at::ScalarType::BFloat16 ? dvm::DType::kFloat32 : k->TransType(result_type);
  auto self_obj = k->Input(self);
  self_obj = k->Cast(self_obj, compute_type);
  if (op_type == dvm::BinaryOpType::kAdd || op_type == dvm::BinaryOpType::kSub) {
    // aclnn add_scalar/sub_scalar round the scalar to the input dtype before
    // the op. Mul/Div keep fp32. Match aclnn here so DVM ON == DVM OFF.
    scalar = RoundScalarToInputDtype(scalar, result_type);
  }
  auto out_obj = k->Binary(op_type, self_obj, scalar);
  if (out == nullptr) {
    k->Output(self, out_obj, true);
  } else {
    if (dst_type == dvm::DType::kBool) {
      result_type = at::ScalarType::Bool;
    }
    *out = k->Output(out_obj, self.sizes(), self.options().dtype(result_type));
  }
  return true;
}

bool AddScalar(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha, at::Tensor *out) {
  auto other_type = other.type();
  auto alpha_type = alpha.type();
  if (other_type == at::ScalarType::Long && alpha_type == at::ScalarType::Long) {
    return BinaryScalar(self, at::Scalar(other.toLong() * alpha.toLong()), dvm::BinaryOpType::kAdd, IsFloatIntType,
                        dvm::DType::kDataTypeEnd, out);
  }
  if ((other_type == at::ScalarType::Long || other_type == at::ScalarType::Double) &&
      (alpha_type == at::ScalarType::Long || alpha_type == at::ScalarType::Double)) {
    double scaled_other = (other_type == at::ScalarType::Long ? static_cast<double>(other.toLong()) : other.toDouble()) *
                          (alpha_type == at::ScalarType::Long ? static_cast<double>(alpha.toLong()) : alpha.toDouble());
    return BinaryScalar(self, at::Scalar(scaled_other), dvm::BinaryOpType::kAdd, IsFloatIntType,
                        dvm::DType::kDataTypeEnd, out);
  }
  return false;
}

bool BinaryTensor(const at::Tensor &self, const at::Tensor &other, dvm::BinaryOpType op_type,
            const std::function<bool(const at::ScalarType &type)> &type_check,
            dvm::DType dst_type = dvm::DType::kDataTypeEnd, at::Tensor *out = nullptr) {
  if (IsCPUScalar(other)) {
    return BinaryScalar(self, other.item(), op_type, type_check, dst_type, out);
  }
  // out == nullptr → inplace into self → self must be contig (DVM Store has no
  // strided form). other is always read-only and may be non-contig.
  // out != nullptr → functional path, both self and other are read-only.
  bool inplace = (out == nullptr);
  if (!InputCheck(self, type_check, /*allow_non_contig=*/!inplace) ||
      !InputCheck(other, type_check, /*allow_non_contig=*/true)) {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
  }
  PrepareFusionInputs(self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  auto result_type = self.scalar_type();
  if (other.scalar_type() != result_type) {
    result_type = at::native::result_type(self, other);
    auto compute_type = result_type == at::ScalarType::BFloat16 ? dvm::DType::kFloat32 : k->TransType(result_type);
    self_obj = k->Cast(self_obj, compute_type);
    other_obj = k->Cast(other_obj, compute_type);
  }
  auto out_obj = k->Binary(op_type, self_obj, other_obj);
  if (out == nullptr) {
    k->Output(self, out_obj, true);
  } else {
    if (dst_type == dvm::DType::kBool) {
      result_type = at::ScalarType::Bool;
    }
    *out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(result_type));
  }
  return true;
}

bool ForeachBinaryScalar(at::TensorList self, const at::Scalar& scalar, dvm::BinaryOpType op_type) {
  for (size_t i = 0; i < self.size(); ++i) {
    if (!InputCheck(self[i])) {
      return false;
    }
  }
  float scalar_value = 1.0f;
  if (!GetScalarValue(scalar, &scalar_value)) {
    return false;
  }
  PrepareWritableOutput(self);
  auto k = g_lazy_fusion_manager.Get();
  for (size_t i = 0; i < self.size(); ++i) {
    auto out_obj = k->Binary(op_type, k->Input(self[i]), scalar_value);
    k->Output(self[i], out_obj, true);
  }
  return true;
}

bool ForeachBinaryScalar(at::TensorList self, at::ArrayRef<at::Scalar> scalars, dvm::BinaryOpType op_type) {
  if (scalars.size() != self.size()) {
    return false;
  }
  std::vector<float> scalar_values(scalars.size());
  for (size_t i = 0; i < self.size(); ++i) {
    if (!InputCheck(self[i]) || !GetScalarValue(scalars[i], &scalar_values[i])) {
      return false;
    }
  }
  PrepareWritableOutput(self);
  auto k = g_lazy_fusion_manager.Get();
  for (size_t i = 0; i < self.size(); ++i) {
    auto out_obj = k->Binary(op_type, k->Input(self[i]), scalar_values[i]);
    k->Output(self[i], out_obj, true);
  }
  return true;
}

bool ForeachAddc(const at::TensorList input, const at::TensorList tensors1,
                 const at::TensorList tensors2, const at::Scalar &scalar, dvm::BinaryOpType op_type) {
  float scalar_value = 1.0f;
  if (tensors1.size() != input.size() || tensors2.size() != input.size() || !GetScalarValue(scalar, &scalar_value)) {
    return false;
  }
  for (size_t i = 0; i < input.size(); ++i) {
    if (!InputCheck(input[i], IsFloat16_32) || !InputCheck(tensors1[i], IsFloat16_32) || !InputCheck(tensors2[i], IsFloat16_32) ||
        tensors1[i].scalar_type() != input[i].scalar_type() || tensors2[i].scalar_type() != input[i].scalar_type()) {
      return false;
    }
  }
  PrepareWritableOutput(input);
  PrepareFusionInputs(tensors1, tensors2);
  auto k = g_lazy_fusion_manager.Get();
  for (size_t i = 0; i < input.size(); ++i) {
    auto t1 = k->Input(input[i]);
    auto t2 = k->Input(tensors1[i]);
    auto t3 = k->Input(tensors2[i]);
    auto out_obj = k->Binary(op_type, t2, t3);
    if (scalar_value != 1.0f) {
      out_obj = k->Binary<dvm::BinaryType::kMul>(out_obj, scalar_value);
    }
    out_obj = k->Binary<dvm::BinaryType::kAdd>(t1, out_obj);
    k->Output(input[i], out_obj, true);
  }
  return true;
}

bool ForeachAddc(const at::TensorList input, const at::TensorList tensors1,
                 const at::TensorList tensors2, at::ArrayRef<at::Scalar> scalars, dvm::BinaryOpType op_type) {
  if (tensors1.size() != input.size() || tensors2.size() != input.size() || scalars.size() != input.size()) {
    return false;
  }
  std::vector<float> scalar_values(scalars.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (!InputCheck(input[i], IsFloat16_32) || !InputCheck(tensors1[i], IsFloat16_32) || !InputCheck(tensors2[i], IsFloat16_32) ||
        tensors1[i].scalar_type() != input[i].scalar_type() || tensors2[i].scalar_type() != input[i].scalar_type() ||
        !GetScalarValue(scalars[i], &scalar_values[i])) {
      return false;
    }
  }
  PrepareWritableOutput(input);
  PrepareFusionInputs(tensors1, tensors2);
  auto k = g_lazy_fusion_manager.Get();
  for (size_t i = 0; i < input.size(); ++i) {
    auto t1 = k->Input(input[i]);
    auto t2 = k->Input(tensors1[i]);
    auto t3 = k->Input(tensors2[i]);
    auto out_obj = k->Binary(op_type, t2, t3);
    if (scalar_values[i] != 1.0f) {
      out_obj = k->Binary<dvm::BinaryType::kMul>(out_obj, scalar_values[i]);
    }
    out_obj = k->Binary<dvm::BinaryType::kAdd>(t1, out_obj);
    k->Output(input[i], out_obj, true);
  }
  return true;
}

struct MatMulAdapter {
  at::Tensor x_tensor;
  at::Tensor y_tensor;
  at::Tensor bias_tensor;
  bool trans_a;
  bool trans_b;
  ShapeVector x_shape;
  ShapeVector y_shape;

  MatMulAdapter(const at::Tensor &x, const at::Tensor &y, bool ta, bool tb, const at::Tensor &bias = at::Tensor())
      : x_tensor(x), y_tensor(y), bias_tensor(bias), trans_a(ta), trans_b(tb) {
    x_shape.assign(x.sizes().begin(), x.sizes().end());
    y_shape.assign(y.sizes().begin(), y.sizes().end());
  }

  bool Check() {
    if (!NpuCheck(x_tensor, y_tensor)) {
      return false;
    }
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(x_tensor) || !at_npu::native::FormatHelper::IsOpInputBaseFormat(y_tensor)) {
      return false;
    }
    auto data_type = x_tensor.scalar_type();
    if (y_tensor.scalar_type() != data_type ||
        (data_type != at::ScalarType::Half && data_type != at::ScalarType::BFloat16)) {
      return false;
    }
    if (bias_tensor.defined() &&
        (!at_npu::native::FormatHelper::IsOpInputBaseFormat(bias_tensor) ||
         !InputCheck(bias_tensor, [&](const at::ScalarType &type) {
           return type == data_type || type == at::ScalarType::Float;
         }))) {
      return false;
    }
    if (x_shape.size() < 2 || x_shape.size() > 4 || y_shape.size() < 2 || y_shape.size() > 4) {
      return false;
    }
    bool check_x = (x_tensor.is_contiguous() || CheckTensorTranspose(x_tensor, &trans_a, &x_shape));
    bool check_y = (y_tensor.is_contiguous() || CheckTensorTranspose(y_tensor, &trans_b, &y_shape));
    if (!CheckMatMulShape()) {
      return false;
    }
    if (!check_x) {
      x_tensor = x_tensor.contiguous();
    }
    if (!check_y) {
      y_tensor = y_tensor.contiguous();
    }
    return true;
  }

 private:
  bool CheckMatMulShape() {
    constexpr int64_t kMaxDimSize = UINT16_MAX - UINT8_MAX;
    return x_shape.back() <= kMaxDimSize && y_shape.back() <= kMaxDimSize;
  }

  bool CheckTensorTranspose(const at::Tensor &tensor, bool *transpose, ShapeVector *shape) {
    const auto &cur_shape = tensor.sizes();
    const auto &cur_strides = tensor.strides();
    int64_t dim1 = cur_shape.size() - 1;
    int64_t dim2 = dim1 - 1;
    if (cur_strides[dim2] == 1 && cur_strides[dim1] == cur_shape[dim2]) {
      int64_t tmpNxD = cur_shape[dim1] * cur_shape[dim2];
      for (int64_t batchDim = dim1 - 2; batchDim >= 0; batchDim--) {
        if (cur_strides[batchDim] != tmpNxD) {
          return false;
        }
        tmpNxD *= cur_shape[batchDim];
      }
      (*transpose) ^= true;
      std::swap((*shape)[dim1], (*shape)[dim2]);
      return true;
    }
    return false;
  }
};

at::Tensor MatMul(const at::Tensor &x_tensor, const at::Tensor &y_tensor, bool trans_a, bool trans_b) {
  MatMulAdapter info(x_tensor, y_tensor, trans_a, trans_b);
  if (!info.Check()) {
    return at::Tensor();
  }
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(info.x_tensor, false, k->GetShapeRef(info.x_shape));
  auto weight_obj = k->Input(info.y_tensor, false, k->GetShapeRef(info.y_shape));
  auto out_obj = k->MatMul(input_obj, weight_obj, info.trans_a, info.trans_b, nullptr);
  return k->Output(out_obj, k->GetShape(out_obj), x_tensor.options().dtype(x_tensor.scalar_type()));
}

template <typename... Args>
void DumpOp(const std::string &op_name, const Args &...inputs) {
  RECORD_FUNCTION(std::string("Dvm::") + op_name, {});
  if (g_lazy_fusion_manager.flags_.dump_as_text) {
    auto k = g_lazy_fusion_manager.Get();
    k->DumpOp(op_name, inputs...);
  }
}
}  // namespace

// ===================== Cast =====================
at::Tensor _npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype) {
  static auto enable = IsEnabled("cast");
  if (!enable || !InputCheck(self, IsSupportType, /*allow_non_contig=*/true) || !IsSupportType(dtype)) {
    return op_api::_npu_dtype_cast(self, dtype);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto self_obj = k->Input(self, false);
  auto dst_type = k->TransType(dtype);
  auto out_obj = k->Cast(self_obj, dst_type);
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(dtype));
  DumpOp("cast", self, dtype);
  return out;
}

// ===================== Unary =====================
at::Tensor abs(const at::Tensor & self) {
  static auto enable = IsEnabled("abs");
  if (!enable || !InputCheck(self, IsFloatIntType, /*allow_non_contig=*/true)) {
    return op_api::abs(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Unary<dvm::UnaryType::kAbs>(k->Input(self));
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("abs", self);
  return out;
}

at::Tensor neg(const at::Tensor & self) {
  static auto enable = IsEnabled("neg");
  if (!enable || !InputCheck(self, IsFloatIntType, /*allow_non_contig=*/true)) {
    return op_api::neg(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Binary<dvm::BinaryType::kMul>(k->Input(self), -1.0f);
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("neg", self);
  return out;
}

at::Tensor sqrt(const at::Tensor & self) {
  static auto enable = IsEnabled("sqrt");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::sqrt(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Unary<dvm::UnaryType::kSqrt>(k->Input(self));
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("sqrt", self);
  return out;
}

at::Tensor exp(const at::Tensor & self) {
  static auto enable = IsEnabled("exp");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::exp(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Unary<dvm::UnaryType::kExp>(k->Input(self));
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("exp", self);
  return out;
}

at::Tensor & exp_(at::Tensor & self) {
  static auto enable = IsEnabled("exp_");
  if (!enable || !InputCheck(self)) {
    return op_api::exp_(self);
  }
  PrepareWritableOutput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Unary<dvm::UnaryType::kExp>(k->Input(self));
  k->Output(self, out_obj, true);
  DumpOp("exp_", self);
  LazyFusionFlush();
  return self;
}

at::Tensor reciprocal(const at::Tensor & self) {
  static auto enable = IsEnabled("reciprocal");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::reciprocal(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Unary<dvm::UnaryType::kReciprocal>(k->Input(self));
  auto out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("reciprocal", self);
  return out;
}

// ===================== Binary: add/sub/mul/div =====================
at::Tensor add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  static auto enable = IsEnabled("add");
  at::Tensor out;
  if (!enable || !AddScalar(self, other, alpha, &out)) {
    return op_api::add(self, other, alpha);
  }
  DumpOp("add", self, other, alpha);
  return out;
}

at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  static auto enable = IsEnabled("add");
  at::Tensor out;
  if (!enable || !Add(self, other, alpha, &out)) {
    return op_api::add(self, other, alpha);
  }
  DumpOp("add", self, other, alpha);
  return out;
}

at::Tensor & add_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  static auto enable = IsEnabled("add_");
  if (!enable || !Add(self, other, alpha)) {
    return op_api::add_(self, other, alpha);
  }
  DumpOp("add_", self, other, alpha);
  LazyFusionFlush();
  return self;
}

at::Tensor sub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  static auto enable = IsEnabled("sub");
  at::Tensor out;
  if (!enable || !Sub(self, other, alpha, &out)) {
    return op_api::sub(self, other, alpha);
  }
  DumpOp("sub", self, other, alpha);
  return out;
}

at::Tensor & sub_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  static auto enable = IsEnabled("sub_");
  if (!enable || !Sub(self, other, alpha)) {
    return op_api::sub_(self, other, alpha);
  }
  DumpOp("sub_", self, other, alpha);
  LazyFusionFlush();
  return self;
}

at::Tensor mul(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("mul");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kMul, IsFloatIntType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::mul(self, other);
  }
  DumpOp("mul", self, other);
  return out;
}

at::Tensor mul(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("mul");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kMul, IsFloatIntType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::mul(self, other);
  }
  DumpOp("mul", self, other);
  return out;
}

at::Tensor & mul_(at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("mul_");
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kMul, IsFloatIntType)) {
    return op_api::mul_(self, other);
  }
  DumpOp("mul_", self, other);
  LazyFusionFlush();
  return self;
}

at::Tensor & mul_(at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("mul_");
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kMul, IsFloatIntType)) {
    return op_api::mul_(self, other);
  }
  DumpOp("mul_", self, other);
  LazyFusionFlush();
  return self;
}

at::Tensor div(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("div");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kDiv, IsFloatType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::div(self, other);
  }
  DumpOp("div", self, other);
  return out;
}

at::Tensor div(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("div");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kDiv, IsFloatType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::div(self, other);
  }
  DumpOp("div", self, other);
  return out;
}

at::Tensor & div_(at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("div_");
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kDiv, IsFloatType)) {
    return op_api::div_(self, other);
  }
  DumpOp("div_", self, other);
  LazyFusionFlush();
  return self;
}

at::Tensor & div_(at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("div_");
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kDiv, IsFloatType)) {
    return op_api::div_(self, other);
  }
  DumpOp("div_", self, other);
  LazyFusionFlush();
  return self;
}

// ===================== Pow =====================
at::Tensor pow(const at::Tensor & self, const at::Scalar & exponent) {
  static auto enable = IsEnabled("pow");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, exponent, dvm::BinaryOpType::kPow, IsFloatType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::pow(self, exponent);
  }
  DumpOp("pow", self, exponent);
  return out;
}

at::Tensor pow(const at::Tensor & self, const at::Tensor & exponent) {
  static auto enable = IsEnabled("pow");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, exponent, dvm::BinaryOpType::kPow, IsFloatType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::pow(self, exponent);
  }
  DumpOp("pow", self, exponent);
  return out;
}

at::Tensor & pow_(at::Tensor & self, const at::Scalar & exponent) {
  static auto enable = IsEnabled("pow_");
  if (!enable || !BinaryScalar(self, exponent, dvm::BinaryOpType::kPow, IsFloatType)) {
    return op_api::pow_(self, exponent);
  }
  DumpOp("pow_", self, exponent);
  LazyFusionFlush();
  return self;
}

at::Tensor & pow_(at::Tensor & self, const at::Tensor & exponent) {
  static auto enable = IsEnabled("pow_");
  if (!enable || !BinaryTensor(self, exponent, dvm::BinaryOpType::kPow, IsFloatType)) {
    return op_api::pow_(self, exponent);
  }
  DumpOp("pow_", self, exponent);
  LazyFusionFlush();
  return self;
}

// ===================== FloorDivide =====================
bool FloorDivideScalar(const at::Tensor &self, const at::Scalar &other, at::Tensor *out) {
  bool inplace = (out == nullptr);
  if (!InputCheck(self, IsFloatType, /*allow_non_contig=*/!inplace)) {
    return false;
  }
  float scalar = 1.0f;
  auto other_type = other.type();
  if (other_type == at::ScalarType::Long) {
    scalar = static_cast<float>(other.toLong());
  } else if (other_type == at::ScalarType::Double) {
    scalar = static_cast<float>(other.toDouble());
  } else {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto compute_type = self.scalar_type() == at::ScalarType::BFloat16
                          ? dvm::DType::kFloat32 : k->TransType(self.scalar_type());
  auto self_obj = k->Input(self);
  self_obj = k->Cast(self_obj, compute_type);
  auto div_obj = k->Binary<dvm::BinaryType::kDiv>(self_obj, scalar);
  auto floor_obj = k->Unary<dvm::UnaryType::kFloor>(div_obj);
  if (out == nullptr) {
    k->Output(self, floor_obj, true);
  } else {
    *out = k->Output(floor_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  }
  return true;
}

bool FloorDivideTensor(const at::Tensor &self, const at::Tensor &other, at::Tensor *out) {
  if (IsCPUScalar(other)) {
    return FloorDivideScalar(self, other.item(), out);
  }
  bool inplace = (out == nullptr);
  if (!InputCheck(self, IsFloatType) || !InputCheck(other, IsFloatType)) {
    return false;
  }
  if (out == nullptr) {
    PrepareWritableOutput(self);
  }
  PrepareFusionInputs(self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  auto result_type = self.scalar_type();
  if (other.scalar_type() != result_type) {
    result_type = at::native::result_type(self, other);
    auto compute_type = result_type == at::ScalarType::BFloat16
                            ? dvm::DType::kFloat32 : k->TransType(result_type);
    self_obj = k->Cast(self_obj, compute_type);
    other_obj = k->Cast(other_obj, compute_type);
  }
  auto div_obj = k->Binary<dvm::BinaryType::kDiv>(self_obj, other_obj);
  auto floor_obj = k->Unary<dvm::UnaryType::kFloor>(div_obj);
  if (out == nullptr) {
    k->Output(self, floor_obj, true);
  } else {
    *out = k->Output(floor_obj, k->GetShape(floor_obj), self.options().dtype(result_type));
  }
  return true;
}

at::Tensor floor_divide(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("floor_divide");
  at::Tensor out;
  if (!enable || !FloorDivideTensor(self, other, &out)) {
    return op_api::floor_divide(self, other);
  }
  DumpOp("floor_divide", self, other);
  return out;
}

at::Tensor floor_divide(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("floor_divide");
  at::Tensor out;
  if (!enable || !FloorDivideScalar(self, other, &out)) {
    return op_api::floor_divide(self, other);
  }
  DumpOp("floor_divide", self, other);
  return out;
}

at::Tensor & floor_divide_(at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("floor_divide_");
  if (!enable || !FloorDivideTensor(self, other, nullptr)) {
    return op_api::floor_divide_(self, other);
  }
  DumpOp("floor_divide_", self, other);
  LazyFusionFlush();
  return self;
}

at::Tensor & floor_divide_(at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("floor_divide_");
  if (!enable || !FloorDivideScalar(self, other, nullptr)) {
    return op_api::floor_divide_(self, other);
  }
  DumpOp("floor_divide_", self, other);
  LazyFusionFlush();
  return self;
}

at::Tensor & floor_divide_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  static auto enable = IsEnabled("floor_divide");
  if (!enable || !NpuCheck(out) || IsCPUScalar(other) ||
      !InputCheck(self, IsFloatType) || !InputCheck(other, IsFloatType)) {
    return op_api::floor_divide_out(self, other, out);
  }
  PrepareWritableOutput(out);
  PrepareFusionInputs(self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  if (self.scalar_type() != other.scalar_type()) {
    auto result_type = at::native::result_type(self, other);
    auto compute_type = result_type == at::ScalarType::BFloat16
                            ? dvm::DType::kFloat32 : k->TransType(result_type);
    self_obj = k->Cast(self_obj, compute_type);
    other_obj = k->Cast(other_obj, compute_type);
  }
  auto div_obj = k->Binary<dvm::BinaryType::kDiv>(self_obj, other_obj);
  auto floor_obj = k->Unary<dvm::UnaryType::kFloor>(div_obj);
  k->Output(out, floor_obj, false);
  DumpOp("floor_divide_out", self, other);
  LazyFusionFlush();
  return out;
}

// ===================== Comparison =====================
at::Tensor eq(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("eq");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::eq(self, other);
  }
  DumpOp("eq", self, other);
  return out;
}

at::Tensor eq(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("eq");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::eq(self, other);
  }
  DumpOp("eq", self, other);
  return out;
}

at::Tensor ne(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("ne");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kNotEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::ne(self, other);
  }
  DumpOp("ne", self, other);
  return out;
}

at::Tensor ne(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("ne");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kNotEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::ne(self, other);
  }
  DumpOp("ne", self, other);
  return out;
}

at::Tensor gt(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("gt");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kGreater, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::gt(self, other);
  }
  DumpOp("gt", self, other);
  return out;
}

at::Tensor gt(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("gt");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kGreater, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::gt(self, other);
  }
  DumpOp("gt", self, other);
  return out;
}

at::Tensor ge(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("ge");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kGreaterEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::ge(self, other);
  }
  DumpOp("ge", self, other);
  return out;
}

at::Tensor ge(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("ge");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kGreaterEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::ge(self, other);
  }
  DumpOp("ge", self, other);
  return out;
}

at::Tensor lt(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("lt");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kLess, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::lt(self, other);
  }
  DumpOp("lt", self, other);
  return out;
}

at::Tensor lt(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("lt");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kLess, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::lt(self, other);
  }
  DumpOp("lt", self, other);
  return out;
}

at::Tensor le(const at::Tensor & self, const at::Scalar & other) {
  static auto enable = IsEnabled("le");
  at::Tensor out;
  if (!enable || !BinaryScalar(self, other, dvm::BinaryOpType::kLessEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::le(self, other);
  }
  DumpOp("le", self, other);
  return out;
}

at::Tensor le(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("le");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kLessEqual, IsFloatType, dvm::DType::kBool, &out)) {
    return op_api::le(self, other);
  }
  DumpOp("le", self, other);
  return out;
}

at::Tensor maximum(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("maximum");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kMaximum, IsFloatIntType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::maximum(self, other);
  }
  DumpOp("maximum", self, other);
  return out;
}

at::Tensor minimum(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("minimum");
  at::Tensor out;
  if (!enable || !BinaryTensor(self, other, dvm::BinaryOpType::kMinimum, IsFloatIntType, dvm::DType::kDataTypeEnd, &out)) {
    return op_api::minimum(self, other);
  }
  DumpOp("minimum", self, other);
  return out;
}

// ===================== Select =====================
at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("where");
  auto cond_ok = condition.defined() && condition.is_contiguous() && torch_npu::utils::is_npu(condition) &&
                 condition.scalar_type() == at::ScalarType::Bool;
  if (!enable || !cond_ok ||
      !InputCheck(self, IsFloatType) || !InputCheck(other, IsFloatType)) {
    return op_api::where(condition, self, other);
  }
  PrepareFusionInputs(condition, self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto cond_obj = k->Input(condition);
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  auto result_type = self.scalar_type();
  if (other.scalar_type() != result_type) {
    result_type = at::native::result_type(self, other);
    auto compute_type = result_type == at::ScalarType::BFloat16 ? dvm::DType::kFloat32 : k->TransType(result_type);
    self_obj = k->Cast(self_obj, compute_type);
    other_obj = k->Cast(other_obj, compute_type);
  }
  auto out_obj = k->Select(cond_obj, self_obj, other_obj);
  auto out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(result_type));
  DumpOp("where", condition, self, other);
  return out;
}

at::Tensor & where_out(const at::Tensor & condition, const at::Tensor & self,
                       const at::Tensor & other, at::Tensor & out) {
  static auto enable = IsEnabled("where");
  auto cond_ok = condition.defined() && condition.is_contiguous() && torch_npu::utils::is_npu(condition) &&
                 condition.scalar_type() == at::ScalarType::Bool;
  if (!enable || !cond_ok ||
      !InputCheck(self, IsFloatType) || !InputCheck(other, IsFloatType) || !NpuCheck(out)) {
    return op_api::where_out(condition, self, other, out);
  }
  PrepareWritableOutput(out);
  PrepareFusionInputs(condition, self, other);
  auto k = g_lazy_fusion_manager.Get();
  auto cond_obj = k->Input(condition);
  auto self_obj = k->Input(self);
  auto other_obj = k->Input(other);
  if (self.scalar_type() != other.scalar_type()) {
    auto result_type = at::native::result_type(self, other);
    auto compute_type = result_type == at::ScalarType::BFloat16 ? dvm::DType::kFloat32 : k->TransType(result_type);
    self_obj = k->Cast(self_obj, compute_type);
    other_obj = k->Cast(other_obj, compute_type);
  }
  auto out_obj = k->Select(cond_obj, self_obj, other_obj);
  k->Output(out, out_obj, false);
  DumpOp("where_out", condition, self, other);
  LazyFusionFlush();
  return out;
}

// ===================== Activation =====================
at::Tensor sigmoid(const at::Tensor & self) {
  static auto enable = IsEnabled("sigmoid");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::sigmoid(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
  auto neg_x = k->Binary<dvm::BinaryType::kMul>(input_obj, -1.0f);
  auto exp_neg_x = k->Unary<dvm::UnaryType::kExp>(neg_x);
  auto add_exp = k->Binary<dvm::BinaryType::kAdd>(exp_neg_x, 1.0f);
  auto obj = k->Unary<dvm::UnaryType::kReciprocal>(add_exp);
  auto out = k->Output(obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("sigmoid", self);
  return out;
}

at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output) {
  static auto enable = IsEnabled("sigmoid_backward");
  if (!enable ||
      !InputCheck(grad_output, IsFloatType, /*allow_non_contig=*/true) ||
      !InputCheck(output, IsFloatType, /*allow_non_contig=*/true) ||
      grad_output.scalar_type() != output.scalar_type()) {
    return op_api::sigmoid_backward(grad_output, output);
  }
  PrepareFusionInputs(grad_output, output);
  auto k = g_lazy_fusion_manager.Get();
  auto y_obj = k->Input(output);
  auto dy_obj = k->Input(grad_output);
  y_obj = k->Cast(y_obj, dvm::DType::kFloat32);
  dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
  auto one_sub_y = k->Binary<dvm::BinaryType::kSub>(1.0f, y_obj);
  auto y_mul_dy = k->Binary<dvm::BinaryType::kMul>(y_obj, dy_obj);
  auto obj = k->Binary<dvm::BinaryType::kMul>(one_sub_y, y_mul_dy);
  auto out = k->Output(obj, k->GetShape(obj), grad_output.options().dtype(grad_output.scalar_type()));
  DumpOp("sigmoid_backward", grad_output, output);
  return out;
}

at::Tensor tanh(const at::Tensor & self) {
  static auto enable = IsEnabled("tanh");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::tanh(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto x = k->Input(self);
  x = k->Cast(x, dvm::DType::kFloat32);
  // Align with the run-package tanh kernel: clip x to [-8.8, 8.8] in float32,
  // then compute tanh(x) = (exp(2x) - 1) / (exp(2x) + 1).
  auto obj = BuildClampedTanhFp32(k, x);
  auto out = k->Output(obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("tanh", self);
  return out;
}

at::Tensor & tanh_(at::Tensor & self) {
  static auto enable = IsEnabled("tanh_");
  if (!enable || !InputCheck(self)) {
    return op_api::tanh_(self);
  }
  PrepareWritableOutput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto x = k->Input(self);
  x = k->Cast(x, dvm::DType::kFloat32);
  auto out_obj = BuildClampedTanhFp32(k, x);
  k->Output(self, out_obj, true);
  DumpOp("tanh_", self);
  LazyFusionFlush();
  return self;
}

at::Tensor tanh_backward(const at::Tensor & grad_output, const at::Tensor & output) {
  static auto enable = IsEnabled("tanh_backward");
  if (!enable ||
      !InputCheck(grad_output, IsFloatType) || !InputCheck(output, IsFloatType) ||
      grad_output.scalar_type() != output.scalar_type()) {
    return op_api::tanh_backward(grad_output, output);
  }
  PrepareFusionInputs(grad_output, output);
  auto k = g_lazy_fusion_manager.Get();
  auto dy = k->Input(grad_output);
  auto y = k->Input(output);
  y = k->Cast(y, dvm::DType::kFloat32);
  dy = k->Cast(dy, dvm::DType::kFloat32);
  // grad = dy * (1 - y * y)
  auto y_sq = k->Binary<dvm::BinaryType::kMul>(y, y);
  auto one_sub_ysq = k->Binary<dvm::BinaryType::kSub>(1.0f, y_sq);
  auto obj = k->Binary<dvm::BinaryType::kMul>(dy, one_sub_ysq);
  auto out = k->Output(obj, k->GetShape(obj), grad_output.options().dtype(grad_output.scalar_type()));
  DumpOp("tanh_backward", grad_output, output);
  return out;
}

at::Tensor & gelu_out(const at::Tensor & self, c10::string_view approximate, at::Tensor & out) {
  static auto enable = IsEnabled("gelu");
  if (!enable || !InputCheck(self, IsFloatType) || !NpuCheck(out) || approximate != "tanh") {
    return op_api::gelu_out(self, approximate, out);
  }
  PrepareWritableOutput(out);
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  constexpr float kGeluCoeffA = 0.044715f;
  constexpr float kGeluCoeffB = 1.5957691216057308f;
  auto x_obj = k->Input(self);
  x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
  auto x_squared = k->Binary<dvm::BinaryType::kMul>(x_obj, x_obj);
  auto x_cubed = k->Binary<dvm::BinaryType::kMul>(x_squared, x_obj);
  auto cubic_term = k->Binary<dvm::BinaryType::kMul>(x_cubed, kGeluCoeffA);
  auto tanh_param = k->Binary<dvm::BinaryType::kAdd>(x_obj, cubic_term);
  auto y = k->Binary<dvm::BinaryType::kMul>(tanh_param, kGeluCoeffB);
  auto exp_min_y_0 = k->Unary<dvm::UnaryType::kExp>(k->Binary<dvm::BinaryType::kMinimum>(y, 0.0f));
  auto exp_neg_abs_y = k->Unary<dvm::UnaryType::kExp>(
      k->Binary<dvm::BinaryType::kMul>(k->Unary<dvm::UnaryType::kAbs>(y), -1.0f));
  auto denom = k->Binary<dvm::BinaryType::kAdd>(exp_neg_abs_y, 1.0f);
  auto div = k->Binary<dvm::BinaryType::kDiv>(x_obj, denom);
  auto obj = k->Binary<dvm::BinaryType::kMul>(div, exp_min_y_0);
  k->Output(out, obj, false);
  DumpOp("gelu", self, approximate);
  return out;
}

at::Tensor gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {
  static auto enable = IsEnabled("gelu_backward");
  if (!enable ||
      !InputCheck(grad_output, IsFloatType) || !InputCheck(self, IsFloatType) ||
      grad_output.scalar_type() != self.scalar_type()) {
    return op_api::gelu_backward(grad_output, self, approximate);
  }
  PrepareFusionInputs(grad_output, self);
  auto k = g_lazy_fusion_manager.Get();
  constexpr float cs_value = 0.044715f;
  constexpr float cs_sqrt_two_div_pi = 0.7978845608028564f;
  constexpr float cs_value_tri = 0.134145f;
  auto dy_obj = k->Input(grad_output);
  auto x_obj = k->Input(self);
  x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
  dy_obj = k->Cast(dy_obj, dvm::DType::kFloat32);
  auto x_squared = k->Binary<dvm::BinaryType::kMul>(x_obj, x_obj);
  auto x_cubed = k->Binary<dvm::BinaryType::kMul>(x_squared, x_obj);
  auto mul_double_mul_tri = k->Binary<dvm::BinaryType::kMul>(cs_value_tri, x_squared);
  auto mul_add_one = k->Binary<dvm::BinaryType::kAdd>(1.0f, mul_double_mul_tri);
  auto mul_right = k->Binary<dvm::BinaryType::kMul>(cs_sqrt_two_div_pi, mul_add_one);
  auto mul_triple_mul_csvalue = k->Binary<dvm::BinaryType::kMul>(cs_value, x_cubed);
  auto mul_add_x = k->Binary<dvm::BinaryType::kAdd>(x_obj, mul_triple_mul_csvalue);
  auto tanh_para = k->Binary<dvm::BinaryType::kMul>(cs_sqrt_two_div_pi, mul_add_x);
  auto tanh_res = BuildClampedTanhFp32(k, tanh_para);
  auto tanh_res_add_one = k->Binary<dvm::BinaryType::kAdd>(1.0f, tanh_res);
  auto half_mul_tanh_res_add_one = k->Binary<dvm::BinaryType::kMul>(0.5f, tanh_res_add_one);
  auto tanh_res_squared = k->Binary<dvm::BinaryType::kMul>(tanh_res, tanh_res);
  auto one_sub_tanh_res_squared = k->Binary<dvm::BinaryType::kSub>(1.0f, tanh_res_squared);
  auto half_mul_x = k->Binary<dvm::BinaryType::kMul>(0.5f, x_obj);
  auto mul_tmp = k->Binary<dvm::BinaryType::kMul>(half_mul_x, one_sub_tanh_res_squared);
  auto mul_final = k->Binary<dvm::BinaryType::kMul>(mul_tmp, mul_right);
  auto result_tmp = k->Binary<dvm::BinaryType::kAdd>(half_mul_tanh_res_add_one, mul_final);
  auto obj = k->Binary<dvm::BinaryType::kMul>(dy_obj, result_tmp);
  auto out = k->Output(obj, k->GetShape(obj), grad_output.options().dtype(grad_output.scalar_type()));
  DumpOp("gelu_backward", grad_output, self, approximate);
  return out;
}

at::Tensor relu(const at::Tensor & self) {
  static auto enable = IsEnabled("relu");
  if (!enable || !InputCheck(self, IsFloatIntType, /*allow_non_contig=*/true)) {
    return op_api::relu(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto obj = k->Binary<dvm::BinaryType::kMaximum>(k->Input(self), 0.0f);
  auto out = k->Output(obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("relu", self);
  return out;
}

at::Tensor & relu_(at::Tensor & self) {
  static auto enable = IsEnabled("relu_");
  if (!enable || !InputCheck(self, IsFloatIntType)) {
    return op_api::relu_(self);
  }
  PrepareWritableOutput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Binary<dvm::BinaryType::kMaximum>(k->Input(self), 0.0f);
  k->Output(self, out_obj, true);
  DumpOp("relu_", self);
  LazyFusionFlush();
  return self;
}

bool LeakyRelu(const at::Tensor &self, const at::Scalar &negative_slope, at::Tensor *out = nullptr) {
  float slope = 0.0f;
  bool inplace = (out == nullptr);
  if (!InputCheck(self, IsFloatType, /*allow_non_contig=*/!inplace) ||
      !GetScalarValue(negative_slope, &slope)) {
    return false;
  }
  if (inplace) {
    PrepareWritableOutput(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  // LeakyReLU:
  //   y = x,                  if x > 0
  //   y = negative_slope * x, otherwise
  // This is lowered as:
  //   max(x, 0) + negative_slope * min(x, 0)
  auto pos_obj = k->Binary<dvm::BinaryType::kMaximum>(input_obj, 0.0f);
  auto neg_obj = k->Binary<dvm::BinaryType::kMinimum>(input_obj, 0.0f);
  auto out_obj = slope == 0.0f ? pos_obj :
    k->Binary<dvm::BinaryType::kAdd>(pos_obj, k->Binary<dvm::BinaryType::kMul>(neg_obj, slope));
  if (out == nullptr) {
    k->Output(self, out_obj, true);
  } else {
    *out = k->Output(out_obj, self.sizes(), self.options().dtype(self.scalar_type()));
  }
  return true;
}

at::Tensor leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope) {
  static auto enable = IsEnabled("leaky_relu");
  at::Tensor out;
  if (!enable || !LeakyRelu(self, negative_slope, &out)) {
    return op_api::leaky_relu(self, negative_slope);
  }
  DumpOp("leaky_relu", self, negative_slope);
  return out;
}

at::Tensor & leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope) {
  static auto enable = IsEnabled("leaky_relu_");
  if (!enable || !LeakyRelu(self, negative_slope)) {
    return op_api::leaky_relu_(self, negative_slope);
  }
  DumpOp("leaky_relu_", self, negative_slope);
  LazyFusionFlush();
  return self;
}

at::Tensor silu(const at::Tensor & self) {
  static auto enable = IsEnabled("silu");
  if (!enable || !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::silu(self);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
  auto neg_x = k->Binary<dvm::BinaryType::kMul>(input_obj, -1.0f);
  auto exp_neg_x = k->Unary<dvm::UnaryType::kExp>(neg_x);
  auto add_exp = k->Binary<dvm::BinaryType::kAdd>(exp_neg_x, 1.0f);
  auto obj = k->Binary<dvm::BinaryType::kDiv>(input_obj, add_exp);
  auto out = k->Output(obj, self.sizes(), self.options().dtype(self.scalar_type()));
  DumpOp("silu", self);
  return out;
}

at::Tensor silu_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  static auto enable = IsEnabled("silu_backward");
  if (!enable ||
      !InputCheck(grad_output, IsFloatType, /*allow_non_contig=*/true) ||
      !InputCheck(self, IsFloatType, /*allow_non_contig=*/true) ||
      grad_output.scalar_type() != self.scalar_type()) {
    return op_api::silu_backward(grad_output, self);
  }
  PrepareFusionInputs(grad_output, self);
  auto k = g_lazy_fusion_manager.Get();
  auto dout_obj = k->Input(grad_output);
  auto x_obj = k->Input(self);
  x_obj = k->Cast(x_obj, dvm::DType::kFloat32);
  dout_obj = k->Cast(dout_obj, dvm::DType::kFloat32);
  auto neg_x = k->Binary<dvm::BinaryType::kMul>(x_obj, -1.0f);
  auto exp_neg_x = k->Unary<dvm::UnaryType::kExp>(neg_x);
  auto add_exp = k->Binary<dvm::BinaryType::kAdd>(exp_neg_x, 1.0f);
  auto sigmod = k->Unary<dvm::UnaryType::kReciprocal>(add_exp);
  auto out = k->Binary<dvm::BinaryType::kDiv>(x_obj, add_exp);
  auto sigmod_out0 = k->Binary<dvm::BinaryType::kAdd>(sigmod, out);
  auto sigmod_out1 = k->Binary<dvm::BinaryType::kMul>(sigmod, out);
  auto sub_res = k->Binary<dvm::BinaryType::kSub>(sigmod_out0, sigmod_out1);
  auto obj = k->Binary<dvm::BinaryType::kMul>(sub_res, dout_obj);
  auto out_tensor = k->Output(obj, k->GetShape(obj), grad_output.options().dtype(grad_output.scalar_type()));
  DumpOp("silu_backward", grad_output, self);
  return out_tensor;
}

// ===================== BatchNorm =====================
bool BatchNormVectorCheck(const c10::optional<at::Tensor> &tensor, int64_t channels) {
  if (!tensor.has_value()) {
    return true;
  }
  const auto &value = tensor.value();
  return value.dim() == 1 && value.numel() == channels && value.scalar_type() == at::ScalarType::Float;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(
  const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias,
  const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var,
  bool training, double momentum, double eps) {
  static auto enable = IsEnabled("native_batch_norm");
  if (!enable || training || input.scalar_type() != at::ScalarType::Float || input.dim() < 2 || input.dim() > 4 ||
      !NpuCheck(input, weight, bias, running_mean, running_var)) {
    return op_api::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
  }
  auto channels = input.size(1);
  if (!BatchNormVectorCheck(weight, channels) || !BatchNormVectorCheck(bias, channels) ||
      !BatchNormVectorCheck(running_mean, channels) || !BatchNormVectorCheck(running_var, channels)) {
    return op_api::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
  }

  auto x = input.contiguous();
  auto weight_tensor = weight.has_value() ? weight.value().contiguous() : at::Tensor();
  auto bias_tensor = bias.has_value() ? bias.value().contiguous() : at::Tensor();
  auto running_mean_tensor = running_mean.has_value() ?
    running_mean.value().contiguous() : at::zeros({channels}, input.options().dtype(at::kFloat));
  auto running_var_tensor = running_var.has_value() ?
    running_var.value().contiguous() : at::ones({channels}, input.options().dtype(at::kFloat));
  auto save_mean = at::zeros({channels}, input.options().dtype(at::kFloat));
  auto save_invstd = at::zeros({channels}, input.options().dtype(at::kFloat));

  // Inference BatchNorm:
  //   y = ((x - running_mean) / sqrt(running_var + eps)) * weight + bias
  // Here invstd is materialized in-graph as:
  //   invstd = rsqrt(running_var + eps)
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto out_obj = k->Input(x);
  auto input_dtype = dvm::Kernel::GetDType(out_obj);
  ShapeVector new_shape(x.dim(), 1);
  new_shape[1] = channels;
  auto new_shape_ref = k->GetShapeRef(new_shape);
  auto mean_obj = k->Cast(k->Input(running_mean_tensor, true, new_shape_ref), input_dtype);
  auto var_obj = k->Cast(k->Input(running_var_tensor, true, new_shape_ref), input_dtype);
  auto invstd_obj =
    k->Unary<dvm::UnaryType::kReciprocal>(
      k->Unary<dvm::UnaryType::kSqrt>(k->Binary<dvm::BinaryType::kAdd>(var_obj, static_cast<float>(eps))));
  out_obj = k->Binary<dvm::BinaryType::kSub>(out_obj, mean_obj);
  out_obj = k->Binary<dvm::BinaryType::kMul>(out_obj, invstd_obj);
  if (weight.has_value()) {
    out_obj = k->Binary<dvm::BinaryType::kMul>(
      out_obj, k->Cast(k->Input(weight_tensor, true, new_shape_ref), input_dtype));
  }
  if (bias.has_value()) {
    out_obj = k->Binary<dvm::BinaryType::kAdd>(
      out_obj, k->Cast(k->Input(bias_tensor, true, new_shape_ref), input_dtype));
  }
  auto out = k->Output(out_obj, k->GetShape(out_obj), input.options().dtype(input.scalar_type()));
  DumpOp("native_batch_norm", input, weight, bias, running_mean, running_var, training, momentum, eps);
  return std::make_tuple(std::move(out), std::move(save_mean), std::move(save_invstd));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(
  const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight,
  const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var,
  const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd,
  bool train, double eps, ::std::array<bool,3> output_mask) {
  static auto enable = IsEnabled("native_batch_norm_backward", Level::kO2);
  if (!enable || train || grad_out.scalar_type() != at::ScalarType::Float || input.scalar_type() != at::ScalarType::Float ||
      grad_out.scalar_type() != input.scalar_type() || input.dim() < 2 || input.dim() > 4 ||
      !NpuCheck(grad_out, input, weight, running_mean, running_var)) {
    return op_api::native_batch_norm_backward(
      grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  }
  auto channels = input.size(1);
  if (!BatchNormVectorCheck(weight, channels) ||
      !BatchNormVectorCheck(running_mean, channels) ||
      !BatchNormVectorCheck(running_var, channels)) {
    return op_api::native_batch_norm_backward(
      grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  }

  auto x = input.contiguous();
  auto dy = grad_out.contiguous();
  auto weight_tensor = weight.has_value() ?
    weight.value().contiguous() : at::ones({channels}, input.options().dtype(at::kFloat));
  auto running_mean_tensor = running_mean.has_value() ?
    running_mean.value().contiguous() : at::zeros({channels}, input.options().dtype(at::kFloat));
  auto running_var_tensor = running_var.has_value() ?
    running_var.value().contiguous() : at::ones({channels}, input.options().dtype(at::kFloat));

  // Inference BatchNorm backward:
  //   grad_input = grad_out * weight / sqrt(running_var + eps)
  //   grad_weight = sum(grad_out * (input - running_mean) / sqrt(running_var + eps))
  //   grad_bias = sum(grad_out)
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto x_obj = k->Input(x);
  auto dy_obj = k->Input(dy);
  auto input_dtype = dvm::Kernel::GetDType(x_obj);
  ShapeVector broadcast_shape(x.dim(), 1);
  broadcast_shape[1] = channels;
  auto broadcast_shape_ref = k->GetShapeRef(broadcast_shape);
  ShapeVector reduce_axis;
  reduce_axis.reserve(x.dim() - 1);
  for (int64_t i = 0; i < x.dim(); ++i) {
    if (i != 1) {
      reduce_axis.push_back(i);
    }
  }
  auto reduce_axis_ref = k->GetShapeRef(reduce_axis);
  auto mean_obj = k->Cast(k->Input(running_mean_tensor, true, broadcast_shape_ref), input_dtype);
  auto var_obj = k->Cast(k->Input(running_var_tensor, true, broadcast_shape_ref), input_dtype);
  auto invstd_obj =
    k->Unary<dvm::UnaryType::kReciprocal>(
      k->Unary<dvm::UnaryType::kSqrt>(k->Binary<dvm::BinaryType::kAdd>(var_obj, static_cast<float>(eps))));
  auto scale_obj =
    k->Binary<dvm::BinaryType::kMul>(
      invstd_obj, k->Cast(k->Input(weight_tensor, true, broadcast_shape_ref), input_dtype));

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (output_mask[0]) {
    auto grad_input_obj = k->Binary<dvm::BinaryType::kMul>(dy_obj, scale_obj);
    grad_input = k->Output(grad_input_obj, k->GetShape(grad_input_obj), input.options().dtype(input.scalar_type()));
  }
  if (output_mask[1]) {
    auto centered_obj = k->Binary<dvm::BinaryType::kSub>(x_obj, mean_obj);
    auto norm_obj = k->Binary<dvm::BinaryType::kMul>(centered_obj, invstd_obj);
    auto grad_weight_obj = k->Reduce<dvm::ReduceType::kSum>(
      k->Binary<dvm::BinaryType::kMul>(dy_obj, norm_obj), reduce_axis_ref, false);
    grad_weight = k->Output(grad_weight_obj, k->GetShape(grad_weight_obj), input.options().dtype(at::kFloat));
  }
  if (output_mask[2]) {
    auto grad_bias_obj = k->Reduce<dvm::ReduceType::kSum>(dy_obj, reduce_axis_ref, false);
    grad_bias = k->Output(grad_bias_obj, k->GetShape(grad_bias_obj), input.options().dtype(at::kFloat));
  }
  DumpOp("native_batch_norm_backward", grad_out, input, weight, running_mean, running_var, train, eps);
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(const at::Tensor & input, double eps) {
  static auto enable = IsEnabled("batch_norm_stats");
  if (!enable || input.scalar_type() != at::ScalarType::Float || !NpuCheck(input)) {
    return op_api::batch_norm_stats(input, eps);
  }
  PrepareFusionInput(input);
  auto ndim = input.dim();
  auto input_shape = input.sizes();
  ShapeVector axis;
  axis.reserve(ndim);
  int64_t count = 1;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i != 1) {
      axis.push_back(i);
      count *= input_shape[i];
    }
  }
  if (count <= 0) {
    return op_api::batch_norm_stats(input, eps);
  }
  auto x = input.contiguous();
  auto k = g_lazy_fusion_manager.Get();
  auto axis_ref = k->GetShapeRef(axis);
  auto input_obj = k->Input(x);
  float coef = 1.0f / static_cast<float>(count);
  auto input_mean =
    k->Reduce<dvm::ReduceType::kSum>(k->Binary<dvm::BinaryType::kMul>(input_obj, coef), axis_ref, true);
  auto input_sub_mean = k->Binary<dvm::BinaryType::kSub>(input_obj, input_mean);
  auto input_var = k->Binary<dvm::BinaryType::kMul>(input_sub_mean, input_sub_mean);
  input_var = k->Reduce<dvm::ReduceType::kSum>(k->Binary<dvm::BinaryType::kMul>(input_var, coef), axis_ref, false);
  auto invstd = k->Unary<dvm::UnaryType::kReciprocal>(
                  k->Unary<dvm::UnaryType::kSqrt>(k->Binary<dvm::BinaryType::kAdd>(input_var, static_cast<float>(eps))));
  auto out1 = k->Output(input_mean, k->GetShape(invstd), input.options().dtype(input.scalar_type()));
  auto out2 = k->Output(invstd, k->GetShape(invstd), input.options().dtype(input.scalar_type()));
  DumpOp("batch_norm_stats", input, eps);
  LazyFusionFlush();
  return std::make_tuple(std::move(out1), std::move(out2));
}

::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts(const at::Tensor & input,
  const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean,
  const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
  static auto enable = IsEnabled("batch_norm_gather_stats_with_counts");
  if (!enable || input.scalar_type() != at::ScalarType::Float || !NpuCheck(input, mean, invstd, counts)) {
    return op_api::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  }
  if ((running_mean.has_value() && !InputCheck(running_mean.value())) ||
      (running_var.has_value() && !InputCheck(running_var.value()))) {
    return op_api::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  }
  auto x = input.contiguous();
  auto mean_all = mean.contiguous();
  auto invstd_all = invstd.contiguous();
  auto counts_tensor = counts.contiguous();
  LazyFusionFlush();
  float momentum_reverse = 1.0f - momentum;
  auto k = g_lazy_fusion_manager.Get();
  ShapeVector count_shape{counts_tensor.numel(), 1};
  ShapeVector counts_axis;
  counts_axis.reserve(count_shape.size());
  for (int64_t i = 0; i < static_cast<int64_t>(count_shape.size()); ++i) {
    counts_axis.push_back(i);
  }
  auto count_axis_ref = k->GetShapeRef(counts_axis);
  ShapeVector mean_axis;
  mean_axis.reserve(mean_all.dim());
  for (int64_t i = 0; i < mean_all.dim() - 1; ++i) {
    mean_axis.push_back(i);
  }
  auto mean_axis_ref = k->GetShapeRef(mean_axis);
  auto x_data_type = x.scalar_type();
  auto x_dtype = k->TransType(x_data_type);
  auto count_all_obj = k->Cast(k->Input(counts_tensor, true, k->GetShapeRef(count_shape)), x_dtype);
  auto global_counts = k->Reduce<dvm::ReduceType::kSum>(count_all_obj, count_axis_ref, false);
  auto mean_all_obj = k->Cast(k->Input(mean_all), x_dtype);
  auto global_mean = k->Reduce<dvm::ReduceType::kSum>(
    k->Binary<dvm::BinaryType::kDiv>(k->Binary<dvm::BinaryType::kMul>(mean_all_obj, count_all_obj), global_counts),
    mean_axis_ref, false);
  auto global_mean_tensor = k->Output(global_mean, k->GetShape(global_mean), x.options().dtype(x.scalar_type()));
  auto mean_sub_all = k->Binary<dvm::BinaryType::kSub>(mean_all_obj, global_mean);
  auto std_all = k->Unary<dvm::UnaryType::kReciprocal>(k->Cast(k->Input(invstd_all), x_dtype));
  auto var_all = k->Binary<dvm::BinaryType::kAdd>(k->Binary<dvm::BinaryType::kMul>(std_all, std_all), static_cast<float>(-eps));
  var_all = k->Binary<dvm::BinaryType::kAdd>(var_all, k->Binary<dvm::BinaryType::kMul>(mean_sub_all, mean_sub_all));
  var_all = k->Binary<dvm::BinaryType::kMul>(var_all, count_all_obj);
  auto global_var_sum = k->Reduce<dvm::ReduceType::kSum>(var_all, mean_axis_ref, false);
  auto global_var = k->Binary<dvm::BinaryType::kDiv>(global_var_sum, global_counts);
  auto global_invstd =
    k->Unary<dvm::UnaryType::kReciprocal>(
      k->Unary<dvm::UnaryType::kSqrt>(k->Binary<dvm::BinaryType::kAdd>(global_var, static_cast<float>(eps))));
  auto global_invstd_tensor = k->Output(global_invstd, k->GetShape(global_invstd), x.options().dtype(x.scalar_type()));
  if (running_mean.has_value()) {
    auto running_mean_tensor = running_mean.value();
    auto running_mean_new = k->Binary<dvm::BinaryType::kAdd>(
      k->Binary<dvm::BinaryType::kMul>(k->Cast(k->Input(running_mean_tensor), x_dtype), momentum_reverse),
      k->Binary<dvm::BinaryType::kMul>(global_mean, static_cast<float>(momentum)));
    k->Output(running_mean_tensor, running_mean_new, true);
  }
  if (running_var.has_value()) {
    auto running_var_tensor = running_var.value();
    auto unbiased_global_var =
      k->Binary<dvm::BinaryType::kDiv>(global_var_sum, k->Binary<dvm::BinaryType::kAdd>(global_counts, -1.0f));
    auto running_var_new = k->Binary<dvm::BinaryType::kAdd>(
      k->Binary<dvm::BinaryType::kMul>(k->Cast(k->Input(running_var_tensor), x_dtype), momentum_reverse),
      k->Binary<dvm::BinaryType::kMul>(unbiased_global_var, static_cast<float>(momentum)));
    k->Output(running_var_tensor, running_var_new, true);
  }
  DumpOp("batch_norm_gather_stats_with_counts", input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  LazyFusionFlush();
  return std::make_tuple(std::move(global_mean_tensor), std::move(global_invstd_tensor));
}

at::Tensor batch_norm_elemt(const at::Tensor & input, const c10::optional<at::Tensor> & weight,
                            const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd,
                            double eps) {
  static auto enable = IsEnabled("batch_norm_elemt");
  if (!enable || input.scalar_type() != at::ScalarType::Float || !NpuCheck(input, mean, invstd, weight, bias)) {
    return op_api::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
  }
  auto x = input.contiguous();
  auto weight_tensor = weight.has_value() ? weight.value().contiguous() : at::Tensor();
  auto bias_tensor = bias.has_value() ? bias.value().contiguous() : at::Tensor();
  auto mean_tensor = mean.contiguous();
  auto invstd_tensor = invstd.contiguous();
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(x);
  auto input_dtype = dvm::Kernel::GetDType(input_obj);
  ShapeVector new_shape(x.dim(), 1);
  new_shape[1] = x.size(1);
  auto new_shape_ref = k->GetShapeRef(new_shape);
  input_obj =
    k->Binary<dvm::BinaryType::kSub>(input_obj, k->Cast(k->Input(mean_tensor, true, new_shape_ref), input_dtype));
  input_obj =
    k->Binary<dvm::BinaryType::kMul>(input_obj, k->Cast(k->Input(invstd_tensor, true, new_shape_ref), input_dtype));
  if (weight.has_value()) {
    input_obj =
      k->Binary<dvm::BinaryType::kMul>(input_obj, k->Cast(k->Input(weight_tensor, true, new_shape_ref), input_dtype));
  }
  if (bias.has_value()) {
    input_obj =
      k->Binary<dvm::BinaryType::kAdd>(input_obj, k->Cast(k->Input(bias_tensor, true, new_shape_ref), input_dtype));
  }
  auto out = k->Output(input_obj, k->GetShape(input_obj), input.options().dtype(input.scalar_type()));
  DumpOp("batch_norm_elemt", input, weight, bias, mean, invstd, eps);
  return out;
}

at::Tensor batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean,
  const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & sum_dy,
  const at::Tensor & sum_dy_xmu, const at::Tensor & count) {
  static auto enable = IsEnabled("batch_norm_backward_elemt");
  if (!enable || input.scalar_type() != at::ScalarType::Float ||
      !weight.has_value() || !NpuCheck(grad_out, input, mean, invstd, weight.value(), sum_dy, sum_dy_xmu, count)) {
    return op_api::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  }
  auto dout_tensor_c = grad_out.contiguous();
  auto input_tensor_c = input.contiguous();
  auto mean_tensor_c = mean.contiguous();
  auto invstd_tensor_c = invstd.contiguous();
  auto weight_tensor_c = weight.value().contiguous();
  auto sumd_dy_tensor_c = sum_dy.contiguous();
  auto sum_dy_xmu_tensor_c = sum_dy_xmu.contiguous();
  auto count_tensor_c = count.contiguous();
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto x_obj = k->Input(input_tensor_c);
  auto x_dtype = dvm::Kernel::GetDType(x_obj);
  ShapeVector new_shape(input_tensor_c.dim(), 1);
  new_shape[1] = input_tensor_c.size(1);
  auto new_shape_ref = k->GetShapeRef(new_shape);
  ShapeVector counts_axis;
  counts_axis.reserve(count_tensor_c.dim());
  for (int64_t i = 0; i < count_tensor_c.dim(); ++i) {
    counts_axis.push_back(i);
  }
  auto count_axis_ref = k->GetShapeRef(counts_axis);
  auto global_counts =
    k->Reduce<dvm::ReduceType::kSum>(k->Cast(k->Input(count_tensor_c), x_dtype), count_axis_ref, false);
  auto invstd_obj = k->Cast(k->Input(invstd_tensor_c, true, new_shape_ref), x_dtype);
  auto invstd_dy_xmu =
    k->Binary<dvm::BinaryType::kMul>(k->Binary<dvm::BinaryType::kMul>(invstd_obj, invstd_obj),
              k->Binary<dvm::BinaryType::kDiv>(k->Cast(k->Input(sum_dy_xmu_tensor_c, true, new_shape_ref), x_dtype),
                        global_counts));
  auto x_sub_mean =
    k->Binary<dvm::BinaryType::kSub>(x_obj, k->Cast(k->Input(mean_tensor_c, true, new_shape_ref), x_dtype));
  auto x_invstd = k->Binary<dvm::BinaryType::kMul>(x_sub_mean, invstd_dy_xmu);
  auto t1 = k->Binary<dvm::BinaryType::kSub>(k->Cast(k->Input(dout_tensor_c), x_dtype),
                      k->Binary<dvm::BinaryType::kDiv>(
                                k->Cast(k->Input(sumd_dy_tensor_c, true, new_shape_ref), x_dtype), global_counts));
  auto t2 = k->Binary<dvm::BinaryType::kSub>(t1, x_invstd);
  auto obj = k->Binary<dvm::BinaryType::kMul>(t2,
    k->Binary<dvm::BinaryType::kMul>(invstd_obj, k->Cast(k->Input(weight_tensor_c, true, new_shape_ref), x_dtype)));
  auto out = k->Output(obj, k->GetShape(obj), input.options().dtype(input.scalar_type()));
  DumpOp("batch_norm_backward_elemt", grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  return out;
}

// ===================== Reduce =====================
at::Tensor sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  static auto enable = IsEnabled("sum", Level::kO2);
  auto out_type = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (!enable || out_type != at::ScalarType::Float ||
      !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::sum(self, dtype);
  }
  PrepareFusionInput(self);
  ShapeVector axis(self.dim());
  for (int64_t i = 0; i < self.dim(); ++i) {
    axis[i] = i;
  }
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
  auto out_obj = k->Reduce<dvm::ReduceType::kSum>(input_obj, k->GetShapeRef(axis), false);
  auto out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(out_type));
  DumpOp("sum", self, dtype);
  LazyFusionFlush();
  return out;
}

at::Tensor sum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  static auto enable = IsEnabled("sum", Level::kO2);
  auto out_type = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (!enable || out_type != at::ScalarType::Float ||
      !InputCheck(self, IsFloatType, /*allow_non_contig=*/true)) {
    return op_api::sum(self, dim, keepdim, dtype);
  }
  PrepareFusionInput(self);
  auto dim_value = dim.value_or(at::IntArrayRef{});
  auto ndim = self.dim();
  ShapeVector axis;
  axis.reserve(ndim);
  for (auto i = dim_value.begin(); i != dim_value.end(); ++i) {
    axis.push_back(*i < 0 ? *i + ndim : *i);
  }
  if (axis.empty()) {
    for (int64_t i = 0; i < self.dim(); ++i) {
      axis.push_back(i);
    }
  }
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
  auto out_obj = k->Reduce<dvm::ReduceType::kSum>(input_obj, k->GetShapeRef(axis), keepdim);
  auto out = k->Output(out_obj, k->GetShape(out_obj), self.options().dtype(out_type));
  DumpOp("sum", self, dim, keepdim, dtype);
  LazyFusionFlush();
  return out;
}

at::Tensor & sum_out(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim,
                     c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  static auto enable = IsEnabled("sum", Level::kO2);
  auto out_type = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (!enable || out_type != at::ScalarType::Float ||
      !InputCheck(self, IsFloatType, /*allow_non_contig=*/true) || !NpuCheck(out)) {
    return op_api::sum_out(self, dim, keepdim, dtype, out);
  }
  PrepareWritableOutput(out);
  PrepareFusionInput(self);
  auto dim_value = dim.value_or(at::IntArrayRef{});
  auto ndim = self.dim();
  ShapeVector axis;
  axis.reserve(ndim);
  for (auto i = dim_value.begin(); i != dim_value.end(); ++i) {
    axis.push_back(*i < 0 ? *i + ndim : *i);
  }
  if (axis.empty()) {
    for (int64_t i = 0; i < self.dim(); ++i) {
      axis.push_back(i);
    }
  }
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(self);
  input_obj = k->Cast(input_obj, dvm::DType::kFloat32);
  auto out_obj = k->Reduce<dvm::ReduceType::kSum>(input_obj, k->GetShapeRef(axis), keepdim);
  k->Output(out, out_obj, false);
  DumpOp("sum_out", self, dim, keepdim, dtype);
  LazyFusionFlush();
  return out;
}

// ===================== MatMul =====================
at::Tensor matmul(const at::Tensor & self, const at::Tensor & other) {
  static auto enable = IsEnabled("matmul", Level::kO2);
  if (!enable) {
    return op_api::matmul(self, other);
  }
  auto out = MatMul(self, other, false, false);
  if (!out.defined()) {
    return op_api::matmul(self, other);
  }
  DumpOp("matmul", self, other);
  return out;
}

at::Tensor mm(const at::Tensor & self, const at::Tensor & mat2) {
  static auto enable = IsEnabled("mm", Level::kO2);
  if (!enable) {
    return op_api::mm(self, mat2);
  }
  auto out = MatMul(self, mat2, false, false);
  if (!out.defined()) {
    return op_api::mm(self, mat2);
  }
  DumpOp("mm", self, mat2);
  return out;
}

at::Tensor bmm(const at::Tensor & self, const at::Tensor & mat2) {
  static auto enable = IsEnabled("bmm", Level::kO2);
  if (!enable) {
    return op_api::bmm(self, mat2);
  }
  auto out = MatMul(self, mat2, false, false);
  if (!out.defined()) {
    return op_api::bmm(self, mat2);
  }
  DumpOp("bmm", self, mat2);
  return out;
}

at::Tensor addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2,
                 const at::Scalar & beta, const at::Scalar & alpha) {
  static auto enable = IsEnabled("addmm", Level::kO2);
  float beta_value = 1.0f;
  float alpha_value = 1.0f;
  if (!enable || !GetScalarValue(beta, &beta_value) || !GetScalarValue(alpha, &alpha_value)) {
    return op_api::addmm(self, mat1, mat2, beta, alpha);
  }
  MatMulAdapter info(mat1, mat2, false, false, self);
  if (!info.Check()) {
    return op_api::addmm(self, mat1, mat2, beta, alpha);
  }
  LazyFusionFlush();
  auto k = g_lazy_fusion_manager.Get();
  auto input_obj = k->Input(info.x_tensor, false, k->GetShapeRef(info.x_shape));
  auto weight_obj = k->Input(info.y_tensor, false, k->GetShapeRef(info.y_shape));
  bool use_bias_fast_path = self.dim() == 1 && beta_value == 1.0f && alpha_value == 1.0f;
  auto bias_obj = use_bias_fast_path ? k->Input(info.bias_tensor, false) : nullptr;
  auto out_obj = k->MatMul(input_obj, weight_obj, info.trans_a, info.trans_b, bias_obj);
  if (!use_bias_fast_path) {
    auto compute_type = dvm::Kernel::GetDType(out_obj);
    if (compute_type == dvm::DType::kBFloat16 || compute_type == dvm::DType::kFloat16) {
      compute_type = dvm::DType::kFloat32;
      out_obj = k->Cast(out_obj, compute_type);
    }
    if (alpha_value != 1.0f) {
      out_obj = k->Binary<dvm::BinaryType::kMul>(out_obj, alpha_value);
    }
    if (beta_value != 0.0f) {
      auto self_obj = k->Input(self);
      if (dvm::Kernel::GetDType(self_obj) != compute_type) {
        self_obj = k->Cast(self_obj, compute_type);
      }
      if (beta_value != 1.0f) {
        self_obj = k->Binary<dvm::BinaryType::kMul>(self_obj, beta_value);
      }
      out_obj = k->Binary<dvm::BinaryType::kAdd>(out_obj, self_obj);
    }
  }
  auto out = k->Output(out_obj, k->GetShape(out_obj), self.options());
  DumpOp("addmm", self, mat1, mat2, beta, alpha);
  return out;
}

// ===================== Foreach =====================
std::vector<at::Tensor> _foreach_sqrt(at::TensorList tensors) {
  static auto enable = IsEnabled("_foreach_sqrt");
  if (!enable) {
    return op_api::_foreach_sqrt(tensors);
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!InputCheck(tensors[i])) {
      return op_api::_foreach_sqrt(tensors);
    }
  }
  PrepareFusionInput(tensors);
  auto k = g_lazy_fusion_manager.Get();
  std::vector<at::Tensor> result(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto out_obj = k->Unary<dvm::UnaryType::kSqrt>(k->Input(tensors[i]));
    result[i] = k->Output(out_obj, tensors[i].sizes(), tensors[i].options().dtype(tensors[i].scalar_type()));
  }
  DumpOp("_foreach_sqrt", tensors);
  return result;
}

void _foreach_sqrt_(at::TensorList tensors) {
  static auto enable = IsEnabled("_foreach_sqrt_");
  if (!enable) {
    op_api::_foreach_sqrt_(tensors);
    return;
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (!InputCheck(tensors[i])) {
      op_api::_foreach_sqrt_(tensors);
      return;
    }
  }
  PrepareWritableOutput(tensors);
  auto k = g_lazy_fusion_manager.Get();
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto out_obj = k->Unary<dvm::UnaryType::kSqrt>(k->Input(tensors[i]));
    k->Output(tensors[i], out_obj, true);
  }
  DumpOp("_foreach_sqrt_", tensors);
  LazyFusionFlush();
}

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
  static auto enable = IsEnabled("_foreach_mul_");
  if (!enable || !ForeachBinaryScalar(self, scalar, dvm::BinaryOpType::kMul)) {
    op_api::_foreach_mul_(self, scalar);
    return;
  }
  DumpOp("_foreach_mul_", self, scalar);
  LazyFusionFlush();
}

void _foreach_add_(const at::TensorList self, const at::Scalar& scalar) {
  static auto enable = IsEnabled("_foreach_add_");
  if (!enable || !ForeachBinaryScalar(self, scalar, dvm::BinaryOpType::kAdd)) {
    op_api::_foreach_add_(self, scalar);
    return;
  }
  DumpOp("_foreach_add_", self, scalar);
  LazyFusionFlush();
}

void _foreach_div_(at::TensorList self, const at::Scalar& scalar) {
  static auto enable = IsEnabled("_foreach_div_");
  if (!enable || !ForeachBinaryScalar(self, scalar, dvm::BinaryOpType::kDiv)) {
    op_api::_foreach_div_(self, scalar);
    return;
  }
  DumpOp("_foreach_div_", self, scalar);
  LazyFusionFlush();
}

void _foreach_div_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  static auto enable = IsEnabled("_foreach_div_");
  if (!enable || !ForeachBinaryScalar(tensors, scalars, dvm::BinaryOpType::kDiv)) {
    op_api::_foreach_div_(tensors, scalars);
    return;
  }
  DumpOp("_foreach_div_", tensors, scalars);
  LazyFusionFlush();
}

void _foreach_addcmul_(const at::TensorList input, const at::TensorList tensors1,
                       const at::TensorList tensors2, const at::Scalar &scalar) {
  static auto enable = IsEnabled("_foreach_addcmul_");
  if (!enable || !ForeachAddc(input, tensors1, tensors2, scalar, dvm::BinaryOpType::kMul)) {
    op_api::_foreach_addcmul_(input, tensors1, tensors2, scalar);
    return;
  }
  DumpOp("_foreach_addcmul_", input, tensors1, tensors2, scalar);
  LazyFusionFlush();
}

void _foreach_addcdiv_(const at::TensorList input, const at::TensorList tensors1,
                       const at::TensorList tensors2, const at::Scalar &scalar) {
  static auto enable = IsEnabled("_foreach_addcdiv_");
  if (!enable || !ForeachAddc(input, tensors1, tensors2, scalar, dvm::BinaryOpType::kDiv)) {
    op_api::_foreach_addcdiv_(input, tensors1, tensors2, scalar);
    return;
  }
  DumpOp("_foreach_addcdiv_", input, tensors1, tensors2, scalar);
  LazyFusionFlush();
}

void _foreach_addcdiv_(const at::TensorList input, const at::TensorList tensors1,
                       const at::TensorList tensors2, at::ArrayRef<at::Scalar> scalars) {
  static auto enable = IsEnabled("_foreach_addcdiv_");
  if (!enable || !ForeachAddc(input, tensors1, tensors2, scalars, dvm::BinaryOpType::kDiv)) {
    op_api::_foreach_addcdiv_(input, tensors1, tensors2, scalars);
    return;
  }
  DumpOp("_foreach_addcdiv_", input, tensors1, tensors2, scalars);
  LazyFusionFlush();
}

// ===================== SwiGLU =====================
at::Tensor npu_swiglu(const at::Tensor &self, int64_t dim) {
  static auto enable = IsEnabled("npu_swiglu", Level::kO2);
  if (!enable || !InputCheck(self)) {
    return op_api::npu_swiglu(self, dim);
  }
  auto ndim = self.dim();
  int64_t real_dim = dim < 0 ? dim + ndim : dim;
  if (real_dim < 0 || real_dim >= ndim || self.size(real_dim) % 2 != 0) {
    return op_api::npu_swiglu(self, dim);
  }
  PrepareFusionInput(self);
  auto k = g_lazy_fusion_manager.Get();

  // chunk(x, 2, dim) => x0 and x1 are views along dim
  // Use ViewInput to load each half directly with stride, avoiding View tensors
  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  int64_t half = self_sizes[real_dim] / 2;
  int64_t dim_stride = self_strides[real_dim];

  auto x0_sizes = self_sizes;
  x0_sizes[real_dim] = half;
  auto x0_strides = self_strides;

  auto x1_sizes = self_sizes;
  x1_sizes[real_dim] = half;
  auto x1_strides = self_strides;

  // Load x0 with stride (view of first half)
  auto x0_shape_ref = k->GetShapeRef(x0_sizes);
  auto x0_stride_ref = k->GetShapeRef(x0_strides);
  auto x0_obj = k->ViewInput(self, self.data_ptr(), x0_shape_ref, x0_stride_ref);

  // Load x1 with stride (view of second half, offset by half * dim_stride * element_size)
  auto elem_size = self.element_size();
  auto x1_data_ptr = static_cast<void *>(static_cast<char *>(self.data_ptr()) + half * dim_stride * elem_size);
  auto x1_shape_ref = k->GetShapeRef(x1_sizes);
  auto x1_stride_ref = k->GetShapeRef(x1_strides);
  auto x1_obj = k->ViewInput(self, x1_data_ptr, x1_shape_ref, x1_stride_ref);

  // aclnn SwiGlu fp16 specialization explicitly casts to fp32 for swish accuracy
  // (CANN .../swiglu_common_impl.h). Match that here so DVM's fp16 path stays
  // bit-exact with aclnn.
  if (self.scalar_type() == at::ScalarType::Half) {
    x0_obj = k->Cast(x0_obj, dvm::DType::kFloat32);
    x1_obj = k->Cast(x1_obj, dvm::DType::kFloat32);
  }
  // silu(x0) = x0 * sigmoid(x0) = x0 / (1 + exp(-x0))
  auto neg_x0 = k->Binary<dvm::BinaryType::kMul>(x0_obj, -1.0f);
  auto exp_neg_x0 = k->Unary<dvm::UnaryType::kExp>(neg_x0);
  auto add_exp = k->Binary<dvm::BinaryType::kAdd>(exp_neg_x0, 1.0f);
  auto silu_x0 = k->Binary<dvm::BinaryType::kDiv>(x0_obj, add_exp);

  // swiglu = silu(x0) * x1
  auto out_obj = k->Binary<dvm::BinaryType::kMul>(silu_x0, x1_obj);

  auto out_shape = self_sizes;
  out_shape[real_dim] = half;
  auto out = k->Output(out_obj, out_shape, self.options().dtype(self.scalar_type()));
  DumpOp("npu_swiglu", self, dim);
  return out;
}
}  // namespace lazy_fusion
