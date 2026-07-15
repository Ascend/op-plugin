// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
namespace {
constexpr int64_t DTYPE_NUM_FOR_QUINT4X2 = static_cast<int64_t>(at::ScalarType::QUInt4x2);
constexpr int64_t INT4_IN_INT32_NUM = 8LL;
constexpr int64_t FP4_IN_UINT8_NUM = 2LL;
constexpr int64_t BLOCK_SIZE_BASE_NUM = 32LL;
constexpr int64_t ALIGN_NUM = 2LL;
constexpr int64_t DEFAULT_SCALE_ALG = 0LL;
constexpr int64_t DEFAULT_AXIS = -1LL;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_rotate_quant(const at::Tensor &x, const at::Tensor &rotation,
    const c10::optional<at::Tensor> &alpha, c10::optional<int64_t> dst_dtype, c10::optional<int64_t> axis,
    c10::optional<c10::string_view> round_mode, c10::optional<int64_t> scale_alg, c10::optional<double> dst_type_max,
    c10::optional<bool> transpose_y) {
    TORCH_CHECK(x.defined(), "Input tensor(x) must be defined" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(rotation.defined(), "Input tensor(rotation) must be defined" + OPS_ERROR(ErrCode::PARAM));
    if (alpha.has_value()) {
        TORCH_CHECK(alpha->defined(), "Input tensor(alpha) must be defined when provided" + OPS_ERROR(ErrCode::PARAM));
    }

    auto dim_num = x.dim();
    int64_t dst_dtype_val = dst_dtype.value_or(static_cast<int64_t>(c10_npu::DType::INT8));
    int64_t axis_val = axis.value_or(DEFAULT_AXIS);
    bool transpose_y_val = transpose_y.value_or(false);

    TORCH_CHECK(!transpose_y_val,
        "In the current CANN version, for aclnnRotateQuant, the parameter transpose_y only supports False. "
        "Please set transpose_y=False." +
            OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(axis_val >= -dim_num && axis_val < dim_num,
        "Param (axis) is out of input dimension range" + OPS_ERROR(ErrCode::PARAM));

    bool is_int4_packed = (dst_dtype_val == DTYPE_NUM_FOR_QUINT4X2);
    aclDataType dst_acl_dtype = is_int4_packed ? aclDataType::ACL_DT_UNDEFINED
                                               : c10_npu::GetAclDataType(dst_dtype_val);
    bool is_fp4 = (dst_acl_dtype == aclDataType::ACL_FLOAT4_E2M1);
    bool is_mx_type = (dst_acl_dtype == aclDataType::ACL_FLOAT4_E2M1 ||
        dst_acl_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        dst_acl_dtype == aclDataType::ACL_FLOAT8_E4M3FN);

    ASCEND_LOGI("[npu_rotate_quant]: Getting aclTensor y dtype by Parameter(dst_dtype): %ld", dst_dtype_val);

    auto output_size = op_infer::array_to_small_vector(x.sizes());
    aclDataType y_acltype;
    at::Tensor output_y;

    if (is_int4_packed) {
        y_acltype = aclDataType::ACL_INT32;
        TORCH_CHECK(output_size[dim_num - 1] % INT4_IN_INT32_NUM == 0,
            "Input shape last dim must be divisible by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        output_size[dim_num - 1] /= INT4_IN_INT32_NUM;
        output_y = npu_preparation::apply_tensor_without_format(output_size, c10::ScalarType::Int);
    } else if (is_fp4) {
        y_acltype = aclDataType::ACL_FLOAT4_E2M1;
        TORCH_CHECK(output_size[dim_num - 1] % FP4_IN_UINT8_NUM == 0,
            "The last dim input shape must be divisible by 2 if "
            "output dtype is torch_npu.float4_e2m1" +
                OPS_ERROR(ErrCode::PARAM));
        output_size[dim_num - 1] /= FP4_IN_UINT8_NUM;
        output_y = npu_preparation::apply_tensor_without_format(output_size, c10::ScalarType::Byte);
    } else {
        y_acltype = dst_acl_dtype;
        TORCH_CHECK(y_acltype != aclDataType::ACL_DT_UNDEFINED, "Unsupported dst_dtype value: ", dst_dtype_val,
            OPS_ERROR(ErrCode::PARAM));
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        TORCH_CHECK(scalar_dtype != at::ScalarType::Undefined,
            "Cannot convert aclDataType to ScalarType for dst_dtype: ", dst_dtype_val, OPS_ERROR(ErrCode::PARAM));
        output_y = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(scalar_dtype));
    }
    ASCEND_LOGI(
        "[npu_rotate_quant]: Setting aclTensor y dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());

    TensorWrapper y_wrapper = {output_y, y_acltype};

    at::Tensor output_scale;
    aclDataType scale_acltype;
    if (is_mx_type) {
        auto mxscale_shape = op_infer::array_to_small_vector(x.sizes());
        mxscale_shape.emplace_back(ALIGN_NUM);
        int64_t axis_change = axis_val < 0 ? axis_val + dim_num : axis_val;
        int64_t dim_size = op_infer::CeilDiv(mxscale_shape[axis_change], BLOCK_SIZE_BASE_NUM);
        dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
        mxscale_shape[axis_change] = dim_size;
        at::ScalarType scale_scalar_type = npu_preparation::convert_to_scalar_type(aclDataType::ACL_FLOAT8_E8M0);
        output_scale = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(scale_scalar_type));
        scale_acltype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        int64_t m = x.size(0);
        output_scale = npu_preparation::apply_tensor_without_format({m}, c10::dtype(c10::ScalarType::Float));
        scale_acltype = aclDataType::ACL_FLOAT;
    }
    TensorWrapper scale_wrapper = {output_scale, scale_acltype};

    const at::Tensor &alpha_real = alpha.value_or(at::Tensor());
    double dst_type_max_val = dst_type_max.value_or(0.0);
    std::string round_mode_str = std::string(round_mode.value_or("rint"));
    char *round_mode_ptr = const_cast<char *>(round_mode_str.data());
    int64_t scale_alg_val = scale_alg.value_or(DEFAULT_SCALE_ALG);

    EXEC_NPU_CMD(aclnnRotateQuant, x, rotation, alpha_real, axis_val, round_mode_ptr, scale_alg_val, dst_type_max_val,
        transpose_y_val, y_wrapper, scale_wrapper);
    return std::tuple<at::Tensor, at::Tensor>(output_y, output_scale);
}
} // namespace op_api
