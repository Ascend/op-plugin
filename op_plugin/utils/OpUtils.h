// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#ifndef OP_PULGIN_UTILS_CALCULATE_OP_UTILS
#define OP_PULGIN_UTILS_CALCULATE_OP_UTILS

#include <ATen/ATen.h>

#include "op_plugin/utils/OpConstants.h"
#include "op_plugin/utils/Export.h"
#include "op_plugin/utils/Version.h"

namespace op_plugin {
namespace utils {
enum ForeachTensorDtypeSupport {
    BASE_DTYPE = 0, // support fp, fp16, bf16
    TO_INT32 = 1,   // support fp, fp16, bf16, int32
    TO_INT = 2      // support fp, fp16, bf16, int32, int8, int64
};

enum ForeachInputType {
    TYPE_TENSOR = 0,     // only tensor type
    TYPE_SCALARLIST = 1, // inclue scalarlist type
    TYPE_SCALAR = 2      // inclue scalar type
};

enum ForeachMappingType {
    MAP_SCALAR_DEFAULT = 0,        // mapping between foreach op tensor and scalar
    MAP_SCALARLIST_DEFAULT = 1,    // mapping between foreach op tensor and scalar_list
    MAP_POW_SCALAR_AND_TENSOR = 2, // mapping between foreach op PowScalarAndTensor tensor and scalar
};

using NameVector = c10::SmallVector<at::Dimname, at::kDimVectorStaticSize>;
OP_PLUGIN_HIDDEN std::string get_reduction_str(int64_t reduction);
OP_PLUGIN_HIDDEN int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr);
OP_PLUGIN_HIDDEN bool is_neox_style(std::string rotary_mode);
OP_PLUGIN_HIDDEN bool is_transpose_last_two_dims(const at::Tensor &tensor);
OP_PLUGIN_HIDDEN bool is_nd_to_nz_on_fly(const at::Tensor &self, const at::Tensor &mat2);
OP_PLUGIN_HIDDEN bool is_scalar_one(const c10::Scalar &scalar);
OP_PLUGIN_HIDDEN float get_scalar_float_value(const c10::Scalar &scalar);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, N> get_dimlist_for_tensor(const at::Tensor &self);
OP_PLUGIN_HIDDEN int64_t complete_pad(int64_t s_size, int64_t p_size, int64_t k_size, int64_t stride);
OP_PLUGIN_HIDDEN c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx);
OP_PLUGIN_HIDDEN at::ScalarType get_divide_result_type(const at::Tensor& self, const at::Tensor& other);
OP_PLUGIN_HIDDEN at::ScalarType get_divide_calculate_type(const at::Tensor& self, const at::Tensor& other);
OP_PLUGIN_HIDDEN at::Tensor get_cast_input(const at::Tensor& self, at::ScalarType calculate_type);
OP_PLUGIN_HIDDEN NameVector compute_names_npu(std::vector<at::Tensor> tensor_list);
OP_PLUGIN_HIDDEN double compute_scale(int64_t input_size, int64_t output_size, double scale);
OP_PLUGIN_HIDDEN bool check_foreach_tensor_dtype_spport_base(at::ScalarType tensorDtype);
OP_PLUGIN_HIDDEN bool check_foreach_tensor_dtype_spport_base_and_int(at::ScalarType tensorDtype);
OP_PLUGIN_HIDDEN bool check_foreach_scalar_dtype_spport(at::ScalarType scalarDtype);
OP_PLUGIN_HIDDEN bool check_foreach_tensor_dtype_spport(at::ScalarType tensorDtype, ForeachTensorDtypeSupport tensorDtypeCategory);
OP_PLUGIN_HIDDEN bool check_dtype_foreach(at::ScalarType tensorDtype, ForeachTensorDtypeSupport tensorDtypeCategory,
                                          ForeachInputType inputType, c10::optional<at::ScalarType> scalarDtype = c10::nullopt,
                                          c10::optional<ForeachMappingType> mapping = c10::nullopt);
OP_PLUGIN_HIDDEN bool check_mapping_between_tensor_and_scalar_list(at::ScalarType tensorDtype, at::ScalarType scalarDtype, ForeachMappingType mapping);
OP_PLUGIN_HIDDEN bool check_mapping_between_tensor_and_scalar(at::ScalarType tensorDtype, at::ScalarType scalarDtype, ForeachMappingType mapping);
OP_PLUGIN_HIDDEN void check_input_same_type_as_parameters(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias);
OP_PLUGIN_HIDDEN void check_input_same_type_as_parameters(const at::Tensor& input, const at::Tensor& weight);
OP_PLUGIN_HIDDEN bool is_gte_cann_version_810rc1();
OP_PLUGIN_HIDDEN bool is_gte_cann_version_820rc1();
OP_PLUGIN_HIDDEN int64_t get_rotary_mode(c10::string_view mode);

}  // namespace utils
}  // namespace op_plugin

#endif  // OP_PULGIN_UTILS_CALCULATE_OP_UTILS
