// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
const int64_t INT4_NUMS_IN_INT32_SPACE = 8;

static std::map<int64_t, at::ScalarType> QUANTIZE_SUPPORT_MAP = {
    {static_cast<int64_t>(at::kQUInt8), at::ScalarType::Byte},
    {static_cast<int64_t>(at::kQInt8), at::ScalarType::Char},
    {static_cast<int64_t>(at::kQInt32), at::ScalarType::Int},
    {static_cast<int64_t>(at::kByte), at::ScalarType::Byte},
    {static_cast<int64_t>(at::kChar), at::ScalarType::Char},
    {static_cast<int64_t>(at::kInt), at::ScalarType::Int},
    {static_cast<int64_t>(at::kFloat8_e4m3fn), at::ScalarType::Float8_e4m3fn},
    {static_cast<int64_t>(at::kFloat8_e5m2), at::ScalarType::Float8_e5m2},
    {static_cast<int64_t>(c10_npu::DType::HIFLOAT8), at::ScalarType::Byte}};
};

at::Tensor npu_quantize_by_kernel(
    const at::Tensor& self,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& zero_points_opt,
    int64_t dtype,
    int64_t axis)
{
    // check if aclnn api implemented
    DO_COMPATIBILITY(aclnnQuantize,
        acl_op::npu_quantize(self, scales, zero_points_opt, dtype, axis));
    // check output datatype supported
    TORCH_CHECK(QUANTIZE_SUPPORT_MAP.find(dtype) != QUANTIZE_SUPPORT_MAP.end(),
        "Param (dtype) must be Int8, UInt8, Int32, HiFloat8, Float8_e4m3fn, Float8_e5m2" + OPS_ERROR(ErrCode::TYPE));
    auto output_shape = op_infer::array_to_small_vector(self.sizes());

    at::ScalarType scalarDtype = QUANTIZE_SUPPORT_MAP[dtype];
    aclDataType yAclType = npu_preparation::convert_to_acl_data_type(scalarDtype);
    if (dtype == static_cast<int64_t>(c10_npu::DType::HIFLOAT8)) {
        yAclType = ACL_HIFLOAT8;
    }
    at::Tensor y = npu_preparation::apply_tensor_without_format(output_shape, self.options().dtype(scalarDtype));
    TensorWrapper y_wrapper = {y, yAclType};
    EXEC_NPU_CMD(aclnnQuantize, self, scales, zero_points_opt, yAclType, axis, y_wrapper);
    return y;
};

at::Tensor npu_quantize_by_ascend_quant(
    const at::Tensor& self,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& zero_points_opt,
    int64_t dtype,
    int64_t axis)
{
    at::ScalarType scalarDtype = at::ScalarType::Undefined;
    aclDataType yAclType = ACL_INT8;
    at::Tensor result;

    if (dtype == static_cast<int64_t>(at::kQInt8)) {
        ASCEND_LOGI("[npu_quantize]: Parameter(dtype) is torch.qint8, setting aclTensor out dtype to: %s",
                    at_npu::native::AclDataTypeToString(aclDataType::ACL_INT8).c_str());
        yAclType = ACL_INT8;
        scalarDtype = at::ScalarType::Char;
    } else if (dtype == static_cast<int64_t>(at::ScalarType::QUInt4x2)) {
        // int4 pack to int32
        ASCEND_LOGI("[npu_quantize]: Parameter(dtype) is torch.quint4x2, setting aclTensor out dtype to: %s",
                    at_npu::native::AclDataTypeToString(aclDataType::ACL_INT32).c_str());
        yAclType = ACL_INT32;
        scalarDtype = at::ScalarType::Int;
    } else {
        ASCEND_LOGI("[npu_quantize]: Getting aclTensor out dtype by Parameter(dtype): %ld", dtype);
        yAclType = c10_npu::GetAclDataType(dtype);
        ASCEND_LOGI("[npu_quantize]: Setting aclTensor out to: %s", at_npu::native::AclDataTypeToString(yAclType).c_str());
        scalarDtype = npu_preparation::convert_to_scalar_type(yAclType);
    }

    if (scalarDtype == at::ScalarType::Int) {
        auto output_shape = op_infer::array_to_small_vector(self.sizes());
        auto x_dim_num = self.dim();
        TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
                    "Input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_shape[x_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
        int64_t npu_format = at_npu::native::custom_ops::get_npu_format(self);
        if (npu_format == ACL_FORMAT_FRACTAL_NZ) {
            result = npu_preparation::apply_tensor_with_format(
                output_shape, self.options().dtype(scalarDtype), ACL_FORMAT_FRACTAL_NZ, true);
        } else {
            result = npu_preparation::apply_tensor_without_format(output_shape, self.options().dtype(scalarDtype));
        }
    } else {
        result = npu_preparation::apply_tensor(self, self.options().dtype(scalarDtype));
    }
    TensorWrapper y_wrapper = {result, yAclType};
    const bool sqrt_mode = false;
    static const bool is_ascend_quant_V3_available = check_aclnn_kernel_available("aclnnAscendQuantV3");
    if (!is_ascend_quant_V3_available) {
        EXEC_NPU_CMD(aclnnAscendQuant, self, scales, zero_points_opt, sqrt_mode, "round", yAclType, y_wrapper);
    } else {
        axis = axis < -1 ? axis : -1;
        EXEC_NPU_CMD(aclnnAscendQuantV3, self, scales, zero_points_opt, sqrt_mode, "round", yAclType, axis, y_wrapper);
    }
    return result;
};

at::Tensor npu_quantize(
    const at::Tensor& self,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& zero_points_opt,
    int64_t dtype,
    int64_t axis,
    bool div_mode)
{
    if (div_mode) {
        return npu_quantize_by_kernel(self, scales, zero_points_opt, dtype, axis);
    }
    return npu_quantize_by_ascend_quant(self, scales, zero_points_opt, dtype, axis);
}
}
