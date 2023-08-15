#ifndef OP_PULGIN_UTILS_CALCULATE_OP_UTILS
#define OP_PULGIN_UTILS_CALCULATE_OP_UTILS

#include <ATen/ATen.h>
#include "op_plugin/utils/OpConstants.h"

namespace op_plugin {
namespace utils {
std::string get_reduction_str(int64_t reduction);
int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr);
bool is_transpose_last_two_dims(const at::Tensor &tensor);
bool is_nd_to_nz_on_fly(const at::Tensor &self, const at::Tensor &mat2);
bool is_scalar_one(const c10::Scalar &scalar);
float get_scalar_float_value(const c10::Scalar &scalar);
c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray);
c10::SmallVector<int64_t, N> get_dimlist_for_tensor(const at::Tensor &self);
int64_t complete_pad(int64_t s_size, int64_t p_size, int64_t k_size, int64_t stride);
c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx);
at::ScalarType get_divide_high_type(const at::Tensor& self, const at::Tensor& other);
}  // namespace utils
}  // namespace op_plugin

#endif  // OP_PULGIN_UTILS_CALCULATE_OP_UTILS
