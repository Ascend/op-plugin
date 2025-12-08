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
constexpr int64_t BLOCK_SIZE_BASE_NUM = 32;
constexpr int64_t BLOCK_SIZE_MAX_NUM = 1024;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_dynamic_mx_quant(
    const at::Tensor &input,
    const int64_t axis,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t block_size,
    c10::optional<int64_t> scale_alg)
{
    at::Tensor y;
    at::Tensor mxscale;
    auto y_shape = op_infer::array_to_small_vector(input.sizes());
    auto mxscale_shape = op_infer::array_to_small_vector(input.sizes());
    mxscale_shape.emplace_back(ALIGN_NUM);

    TORCH_CHECK(axis >= -1 * input.dim() && axis < input.dim(),
        "Param (axis) is out of input dimension range" + OPS_ERROR(ErrCode::PARAM));
    int64_t axis_change = axis < 0 ? axis + input.dim() : axis;
    TORCH_CHECK(block_size % BLOCK_SIZE_BASE_NUM == 0 && block_size > 0 && block_size <= BLOCK_SIZE_MAX_NUM,
        "Param (block_size) must be divisible by 32 and no greater than 1024, greater than 0" + OPS_ERROR(ErrCode::PARAM));
    int64_t dim_size = op_infer::CeilDiv(mxscale_shape[axis_change], block_size);
    dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    mxscale_shape[axis_change] = dim_size;
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // prepare for empty output tensor
    aclDataType y_acltype;
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    ASCEND_LOGI("[npu_dynamic_mx_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);
    if (special_output_type) {
        int64_t last_dim_val = y_shape[input.dim() - 1];
        TORCH_CHECK(last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "output dtype is torch_npu.float4_e2m1 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
        y_shape[input.dim() - 1] = last_dim_val / FP4_IN_UINT8_NUM;
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }
    ASCEND_LOGI("[npu_dynamic_mx_quant]: Setting aclTensor y dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());

    mxscale = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale, aclDataType::ACL_FLOAT8_E8M0};
    int64_t scale_alg_optional = scale_alg.has_value() ? scale_alg.value() : DEFAULT_SCALE_ALG;
    EXEC_NPU_CMD(aclnnDynamicMxQuant, input, axis, round_mode_ptr, y_acltype, block_size, scale_alg_optional, y_wrapper, mxscale_wrapper);
    
    return std::make_tuple(y, mxscale);
}

} // namespace op_api