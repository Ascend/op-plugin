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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static std::map<at::ScalarType, aclDataType> ANTI_QUANT_SUPPORT_MAP = {
    {at::ScalarType::Byte, aclDataType::ACL_HIFLOAT8},
    {at::ScalarType::Char, aclDataType::ACL_INT8},
    {at::ScalarType::Int, aclDataType::ACL_INT4},
    {at::ScalarType::Float8_e4m3fn, aclDataType::ACL_FLOAT8_E4M3FN},
    {at::ScalarType::Float8_e5m2, aclDataType::ACL_FLOAT8_E5M2}
};

at::Tensor apply_anti_quant_out_tensor(const at::Tensor &x, at::ScalarType dst_type)
{
    if (x.dtype() == at::ScalarType::Int) {
        auto x_shape = op_infer::array_to_small_vector(x.sizes());
        size_t dim_num = x_shape.size();
        if (dim_num == 0) {
            TORCH_CHECK(false, "No supported for x is scalar when x dtype is int32 " + OPS_ERROR(ErrCode::TYPE));
        }

        x_shape[dim_num - 1] = x_shape[dim_num - 1] * 8;
        return at_npu::native::OpPreparation::apply_tensor(x_shape, x.options().dtype(dst_type), x);
    }

    return at_npu::native::OpPreparation::apply_tensor(x, x.options().dtype(dst_type));
}

at::Tensor npu_anti_quant(const at::Tensor &x, const at::Tensor &scale, const c10::optional<at::Tensor> &offset,
                          c10::optional<int64_t> dst_dtype, c10::optional<int64_t> src_dtype)
{
    auto input_dtype = x.scalar_type();
    if (ANTI_QUANT_SUPPORT_MAP.find(input_dtype) != ANTI_QUANT_SUPPORT_MAP.end()) {
        if (src_dtype.has_value()) {
            if (input_dtype == at::ScalarType::Int && src_dtype != static_cast<int64_t>(at::ScalarType::QUInt4x2) && c10_npu::GetAclDataType(src_dtype.value()) != aclDataType::ACL_INT4) {
                TORCH_CHECK(false, "The datatype of x is int32, src_dtype must be int4 " + OPS_ERROR(ErrCode::TYPE));
            } else if (input_dtype != at::ScalarType::Int && c10_npu::GetAclDataType(src_dtype.value()) != ANTI_QUANT_SUPPORT_MAP[input_dtype]) {
                TORCH_CHECK(false, "The datatype of x must match src_dtype " + OPS_ERROR(ErrCode::TYPE));
            }
        }
    } else {
        TORCH_CHECK(false, "Input x must be int8, int32, hifloat8, float8_e5m2 or float8_e4m3fn " + OPS_ERROR(ErrCode::TYPE));
    }

    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    if (input_dtype == at::ScalarType::Byte || input_dtype == at::ScalarType::Float8_e4m3fn || input_dtype == at::ScalarType::Float8_e5m2) {
        if (scale.scalar_type() != at::ScalarType::Float || (offset.has_value() && offset_real.scalar_type() != at::ScalarType::Float)) {
            TORCH_CHECK(false, "When x datatype is hifloat8, float8_e5m2 or float8_e4m3fn, scale_dtype and offset_dtype is only support float " + OPS_ERROR(ErrCode::TYPE));
        }
    }

    int64_t dst_type = c10::value_or_else(dst_dtype, [] {return 5;});
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    at::ScalarType dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    TORCH_CHECK(c10::toString(dtype) != "UNKNOWN_SCALAR", "Input dst_type must be valid, but got ",
                c10::toString(dtype), OPS_ERROR(ErrCode::TYPE));

    // construct the output tensor of the NPU
    at::Tensor result = apply_anti_quant_out_tensor(x, dtype);

    bool sqrt_mode = false;
    if (input_dtype == at::ScalarType::Byte) {
        TensorWrapper x_wrapper = {x, aclDataType::ACL_HIFLOAT8};
        EXEC_NPU_CMD(aclnnAscendAntiQuant, x_wrapper, scale, offset, y_acltype, sqrt_mode, result);
    } else {
        EXEC_NPU_CMD(aclnnAscendAntiQuant, x, scale, offset, y_acltype, sqrt_mode, result);
    }

    return result;
}
} // namespace op_api
