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

#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/mirror/NPUTypeProperties.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_plugin {
namespace utils {
bool is_neox_style(std::string rotary_mode)
{
    TORCH_CHECK(rotary_mode != "half" || rotary_mode != "interleave",
        "rotary_mode only support half or interleave", OPS_ERROR(ErrCode::VALUE));
    if (rotary_mode == "half") {
        return true;
    } else {
        return false;
    }
}

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

int64_t get_rotary_mode(c10::string_view mode)
{
    if (mode == "half") {
        // ROTATE_HALF模式对应输入为0
        return 0;
    } else if (mode == "interleave") {
        // ROTATE_INTERLEAVED模式对应输入为1
        return 1;
    } else if (mode == "quarter") {
        return 2;
    } else if (mode == "interleave-half") {
        return 3;
    }
}

int64_t make_warp_dim(int64_t dim, int64_t dim_post_expr)
{
    if (dim_post_expr <= 0) {
        dim_post_expr = 1; // this will make range [-1, 0]
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    return dim;
}

bool is_transpose_last_two_dims(const at::Tensor &tensor)
{
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
        tensor_base_size.size() == static_cast<uint64_t>(tensor.dim())) {
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
            (self_inner_axis < kInnerAxisMinLimit && !(static_cast<uint64_t>(self_inner_axis) & 0xF))) &&
           ((mat2_inner_axis >= kInnerAxisMinLimit && mat2_inner_axis <= kInnerAxisMaxLimit) ||
            (mat2_inner_axis < kInnerAxisMinLimit && !(static_cast<uint64_t>(mat2_inner_axis) & 0xF)));
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
        value = static_cast<float>(scalar.toInt());
    }
    return value;
}

c10::SmallVector<int64_t, N> convert_array_to_vector(c10::IntArrayRef intArray)
{
    c10::SmallVector<int64_t, N> intVec;
    for (uint64_t i = 0; i < intArray.size(); i++) {
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
    TORCH_CHECK(stride != 0, "CompletePad stride is zero!", OPS_ERROR(ErrCode::VALUE));
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
    TORCH_CHECK(scales->size() > idx, "idx", idx, "is overrange scales->at(idx) ", scales->size(),
        OPS_ERROR(ErrCode::VALUE));
    return scales->at(idx);
}

at::ScalarType get_divide_result_type(const at::Tensor& self, const at::Tensor& other)
{
    at::ScalarType high_type = at::native::result_type(self, other);
    if (isIntegralType(high_type, true)) {
        high_type = at::kFloat;
    }
    return high_type;
}

at::ScalarType get_divide_calculate_type(const at::Tensor &self, const at::Tensor &other)
{
    at::ScalarType calculate_type = at_npu::native::result_type(self.scalar_type(), other.scalar_type());
    if (isIntegralType(calculate_type, true) || calculate_type == at::kDouble) {
        calculate_type = at::kFloat;
    }
    return calculate_type;
}

at::Tensor get_cast_input(const at::Tensor& self, at::ScalarType calculate_type)
{
    at::Tensor self_cast = (self.dtype() == calculate_type) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, calculate_type);
    self_cast = at_npu::native::OpPreparation::CastBackToOriFormat(self_cast);
    return self_cast;
}

NameVector compute_names_npu(std::vector<at::Tensor> tensor_list)
{
    NameVector names;
    bool has_names = false;

    for (auto tensor : tensor_list) {
        if (tensor.has_names()) {
            has_names = true;
            break;
        }
    }

    if (!has_names) {
        return names;
    }

    for (auto tensor : tensor_list) {
        if (names.empty()) {
            names = tensor.names();
        } else {
            names = NameVector(at::unify_from_right(names, tensor.names()));
        }
    }
    return names;
}

double compute_scale(int64_t input_size, int64_t output_size, double scale)
{
    if (scale > 0.0) {
        return 1.0 / scale ;
    } else {
        return output_size != 0 ? static_cast<double>(input_size) / output_size : 0;
    }
}

bool check_dtype_foreach(at::ScalarType tensorDtype, ForeachTensorDtypeSupport tensorDtypeCategory, ForeachInputType inputType,
                         c10::optional<at::ScalarType> scalarDtype, c10::optional<ForeachMappingType> mapping)
{
    bool result = false;

    // check tensor dtype
    result = check_foreach_tensor_dtype_spport(tensorDtype, tensorDtypeCategory);
    if (!result) {
        return false;
    }

    // check scalr (scalarlist) parm
    at::ScalarType dtype;
    ForeachMappingType mappingType;
    if (scalarDtype == c10::nullopt && mapping == c10::nullopt) {
        return result;
    } else if (scalarDtype != c10::nullopt && mapping != c10::nullopt) {
        dtype = scalarDtype.value();
        mappingType = mapping.value();
    } else {
        TORCH_CHECK(false, "Invalid  scalarType Parm or ForeachMappingType Parm!", OPS_ERROR(ErrCode::PARAM));
    }

    // checke mapping
    switch (inputType) {
        case ForeachInputType::TYPE_SCALAR:
            return check_mapping_between_tensor_and_scalar(tensorDtype, dtype, mappingType);
        case ForeachInputType::TYPE_SCALARLIST:
            return check_mapping_between_tensor_and_scalar_list(tensorDtype, dtype, mappingType);
        case ForeachInputType::TYPE_TENSOR:
            return true;
        default:
            TORCH_CHECK(false, "Invalid inputType Parm!", OPS_ERROR(ErrCode::PARAM));
    }
}

bool check_foreach_tensor_dtype_spport(at::ScalarType tensorDtype, ForeachTensorDtypeSupport tensorDtypeCategory)
{
    // check tensor dtype
    switch (tensorDtypeCategory) {
        case ForeachTensorDtypeSupport::BASE_DTYPE:
            return check_foreach_tensor_dtype_spport_base(tensorDtype);
        case ForeachTensorDtypeSupport::TO_INT32:
            return check_foreach_tensor_dtype_spport_base(tensorDtype) || (tensorDtype == at::ScalarType::Int);
        case ForeachTensorDtypeSupport::TO_INT:
            return check_foreach_tensor_dtype_spport_base_and_int(tensorDtype);
        default:
            TORCH_CHECK(false, "Invalid  ForeachTensorDtypeSupport Parm", OPS_ERROR(ErrCode::PARAM));
    }
}

bool check_foreach_tensor_dtype_spport_base(at::ScalarType tensorDtype)
{
    return (tensorDtype == at::ScalarType::Half || tensorDtype == at::ScalarType::Float ||
            tensorDtype == at::ScalarType::BFloat16);
}

bool check_foreach_tensor_dtype_spport_base_and_int(at::ScalarType tensorDtype)
{
    return (tensorDtype == at::ScalarType::Half || tensorDtype == at::ScalarType::Float ||
            tensorDtype == at::ScalarType::BFloat16 || tensorDtype == at::ScalarType::Int ||
            tensorDtype == at::ScalarType::Char || tensorDtype == at::ScalarType::Long);
}

bool check_foreach_scalar_dtype_spport(at::ScalarType scalarDtype)
{
    return at::isIntegralType(scalarDtype) || at::isFloatingType(scalarDtype);
}

bool check_mapping_between_tensor_and_scalar_list(at::ScalarType tensorDtype, at::ScalarType scalarDtype, ForeachMappingType mapping)
{
    if (!check_foreach_scalar_dtype_spport(scalarDtype)) {
        return false;
    }

    switch (mapping) {
        case ForeachMappingType::MAP_SCALARLIST_DEFAULT:
            return (at::isIntegralType(scalarDtype) && at::isIntegralType(tensorDtype)) ||
                   (at::isFloatingType(scalarDtype) && at::isFloatingType(tensorDtype));
        default:
            TORCH_CHECK(false, "Invalid  ForeachMappingType Parm Between Tensor And ScalarList", OPS_ERROR(ErrCode::PARAM));
    }
}

bool check_mapping_between_tensor_and_scalar(at::ScalarType tensorDtype, at::ScalarType scalarDtype, ForeachMappingType mapping)
{
    if (!check_foreach_scalar_dtype_spport(scalarDtype)) {
        return false;
    }

    switch (mapping) {
        case ForeachMappingType::MAP_SCALAR_DEFAULT:
            return !at::isIntegralType(tensorDtype) && at::isFloatingType(scalarDtype);
        case ForeachMappingType::MAP_POW_SCALAR_AND_TENSOR:
            return true;
        default:
            TORCH_CHECK(false, "Invalid ForeachMappingType Parm Between Tensor And Scalar!", OPS_ERROR(ErrCode::PARAM));
    }
}

void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias)
{
    TORCH_CHECK(input.options().type_equal(weight.options()),
        "Input type (", input.toString(), ") and weight type (", weight.toString(),
        ") should be the same" + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
        "Input type (", input.toString(), ") and bias type (", bias.toString(),
        ") should be the same" + OPS_ERROR(ErrCode::TYPE));
}

void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight)
{
    check_input_same_type_as_parameters(input, weight, at::Tensor());
}

bool is_gte_cann_version_810rc1()
{
    const static bool is_support_inf_norm = []() -> bool {
        return IsGteCANNVersion("8.1.RC1", "CANN");
    }();
    return is_support_inf_norm;
}

bool is_gte_cann_version_820rc1()
{
    const static bool result = IsGteCANNVersion("8.2.RC1", "CANN");
    return result;
}

}  // namespace utils
}  // namespace op_plugin
