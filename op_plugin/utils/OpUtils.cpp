#include "op_plugin/utils/OpUtils.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace op_plugin {
namespace utils {

std::string get_reduction_str(int64_t reduction)
{
  std::string reductionStr;
  if (reduction == at::Reduction::None) {
    reductionStr = "none";
  } else if (reduction == at::Reduction::Mean) {
    reductionStr = "mean";
  } else {
    reductionStr = "sum";
  }
  return reductionStr;
}

int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr)
{
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;  // this will make range [-1, 0]
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

bool is_transpose_last_two_dims(const at::Tensor &tensor) {
  if (tensor.dim() < 2 || tensor.dim() > 3) {
    return false;
  }
  int64_t numel = at_npu::native::NPUNativeFunctions::get_storage_size(tensor);
  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  c10::SmallVector<int64_t, 5> tensor_base_size = at_npu::native::OpPreparation::get_tensor_desc_base_sizes(tensor);
  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
      tensor.size(dim1) == tensor_base_size[dim2] &&
      tensor.size(dim2) == tensor_base_size[dim1] &&
      tensor.numel() == numel &&
      tensor_base_size.size() == tensor.dim()) {
    return true;
  } else {
    return false;
  }
}

bool is_nd_to_nz_on_fly(const at::Tensor &self, const at::Tensor &mat2)
{
  const static int64_t kInnerAxisMinLimit = 128;
  const static int64_t kInnerAxisMaxLimit = 65535;
  if (self.dim() < 2 || mat2.dim() < 2) {
    return false;
  }
  // get inner axis of input after transpose.
  int64_t self_inner_axis = self.size(self.dim() - 1);
  int64_t self_outer_axis = self.size(self.dim() - 2);
  int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
  int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
  if (is_transpose_last_two_dims(self)) {
    self_inner_axis = self.size(self.dim() - 2);
    self_outer_axis = self.size(self.dim() - 1);
  }
  if (is_transpose_last_two_dims(mat2)) {
    mat2_inner_axis = mat2.size(mat2.dim() - 2);
    mat2_outer_axis = mat2.size(mat2.dim() - 1);
  }
  if (self_inner_axis * self_outer_axis <= kInnerAxisMaxLimit &&
      mat2_inner_axis * mat2_outer_axis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return true;
  }
  // self inner_axis and mat2_inner_axis both in [128, 65535] or in (0, 128) and is multi of 16
  return ((self_inner_axis >= kInnerAxisMinLimit && self_inner_axis <= kInnerAxisMaxLimit) ||
          (self_inner_axis < kInnerAxisMinLimit && !(self_inner_axis & 0xF))) &&
         ((mat2_inner_axis >= kInnerAxisMinLimit && mat2_inner_axis <= kInnerAxisMaxLimit) ||
          (mat2_inner_axis < kInnerAxisMinLimit && !(mat2_inner_axis & 0xF)));
}

bool is_scalar_one(const c10::Scalar &scalar)
{
  if (scalar.isIntegral(false)) {
    return scalar.toInt() == 1;
  } else if (scalar.isFloatingPoint()) {
    return fabs(scalar.toFloat() - 1.0) < 1e-6;
  } else {
    return false;
  }
}

float get_scalar_float_value(const c10::Scalar &scalar)
{
  float value;
  if (scalar.isFloatingPoint()) {
    value = scalar.toFloat();
  } else {
    value = (float)scalar.toInt();
  }
  return value;
}

c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray)
{
  c10::SmallVector<int64_t, N> intVec;
  for (int i = 0; i < intArray.size(); i++) {
    intVec.emplace_back(intArray[i]);
  }
  return intVec;
}

c10::SmallVector<int64_t, N> get_dimlist_for_tensor(const at::Tensor &self)
{
  c10::SmallVector<int64_t, N> dimList = {};
  for (int64_t i = 0; i < self.dim(); i++) {
    dimList.emplace_back(i);
  }
  return dimList;
}

int64_t complete_pad(int64_t s_size, int64_t p_size, int64_t k_size, int64_t stride)
{
  int64_t needpads = 0;
  int64_t sizeP = s_size + p_size * 2;
  int64_t leftLen = sizeP - k_size;
  TORCH_CHECK(stride != 0, "CompletePad stride is zero!");
  auto reminder = leftLen % stride;
  if (reminder != 0) {
    needpads = stride - reminder;
  }
  return needpads;
}

c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx)
{
  if (!scales) {
    return c10::nullopt;
  }
  return scales->at(idx);
}

at::ScalarType get_divide_high_type(const at::Tensor& self, const at::Tensor& other) {
  at::ScalarType high_type = at::native::result_type(self, other);
  if (isIntegralType(high_type, true)) {
    high_type = at::kFloat;
  }
  return high_type;
}
}  // namespace utils
}  // namespace op_plugin