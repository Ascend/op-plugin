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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t X_DIM_NUM_MIN = 1;
constexpr int64_t X_DIM_NUM_MAX = 7;
constexpr int64_t MXSCALE_DIM_NUM_MIN = 2;
constexpr int64_t MXSCALE_DIM_NUM_MAX = 8;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t DEFAULT_AXIS = -1;
constexpr int64_t DEFAULT_DST_TYPE = 15;
constexpr int64_t DEFAULT_SRC_TYPE = 296;
constexpr int64_t NUM_DIFFER = 268;
}; // namespace

at::Tensor npu_anti_mx_quant(
    const at::Tensor &x,
    const at::Tensor &mxscale,
    c10::optional<int64_t> axis,
    c10::optional<int64_t> dst_type,
    c10::optional<int64_t> src_type)
{
    int64_t axis_value = axis.has_value() ? axis.value() : DEFAULT_AXIS;
    int64_t dst_type_value = dst_type.has_value() ? dst_type.value() : DEFAULT_DST_TYPE;
    int64_t src_type_value = src_type.has_value() ? src_type.value() : DEFAULT_SRC_TYPE;

    // input x and mxscale dim check
    TORCH_CHECK(x.sizes().size() >= X_DIM_NUM_MIN && x.sizes().size() <= X_DIM_NUM_MAX,
                "X dimNum should be between 1 and 7, got ", x.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(mxscale.sizes().size() >= MXSCALE_DIM_NUM_MIN && mxscale.sizes().size() <= MXSCALE_DIM_NUM_MAX,
                "Mxscale dimNum should be between 2 and 8, got ", mxscale.sizes().size(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(axis_value >= -1 * x.dim() && axis_value < x.dim(),
        "Param (axis) is out of x dimension range" + OPS_ERROR(ErrCode::PARAM));
    
    at::Tensor y;
    auto y_shape = op_infer::array_to_small_vector(x.sizes());

    bool special_input_type = (c10_npu::GetAclDataType(src_type_value) == aclDataType::ACL_FLOAT4_E2M1 ||
                               c10_npu::GetAclDataType(src_type_value) == aclDataType::ACL_FLOAT4_E1M2);
    if (special_input_type) {
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte,
            "When src_type is float4, x dtype must be uint8" + OPS_ERROR(ErrCode::PARAM));
        int64_t last_dim_val = x.size(x.dim() - 1);
        y_shape[x.dim() - 1] = last_dim_val * 2;
    } else {
        TORCH_CHECK(((static_cast<int64_t>(x.scalar_type()) == static_cast<int64_t>(c10::ScalarType(src_type_value))) ||
            (static_cast<int64_t>(x.scalar_type()) + NUM_DIFFER == src_type_value)),
            "For float8, x dtype must be same as src_type, please check" + OPS_ERROR(ErrCode::PARAM));
    }

    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type_value);
    at::ScalarType dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    y = npu_preparation::apply_tensor_without_format(y_shape, dtype);

    aclDataType x_acltype = aclDataType::ACL_FLOAT8_E4M3FN;
    if (c10_npu::GetAclDataType(src_type_value) == aclDataType::ACL_FLOAT4_E2M1) {
        x_acltype = aclDataType::ACL_FLOAT4_E2M1;
    } else if (c10_npu::GetAclDataType(src_type_value) == aclDataType::ACL_FLOAT4_E1M2) {
        x_acltype = aclDataType::ACL_FLOAT4_E1M2;
    } else if (c10_npu::GetAclDataType(src_type_value) == aclDataType::ACL_FLOAT8_E5M2) {
        x_acltype = aclDataType::ACL_FLOAT8_E5M2;
    }

    ASCEND_LOGI("[npu_anti_mx_quant]: Setting aclTensor y dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());

    TensorWrapper x_wrapper = {x, x_acltype};
    TensorWrapper mxscale_wrapper = {mxscale, aclDataType::ACL_FLOAT8_E8M0};
    EXEC_NPU_CMD(aclnnAntiMxQuant, x_wrapper, mxscale_wrapper, axis_value, y_acltype, y);
    
    return y;
}

} // namespace op_api