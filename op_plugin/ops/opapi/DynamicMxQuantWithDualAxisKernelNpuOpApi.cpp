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
constexpr int64_t MIN_INPUT_DIM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_mx_quant_with_dual_axis(
    const at::Tensor &input,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t scale_alg)
{
    at::Tensor y1;
    at::Tensor mxscale1;
    at::Tensor y2;
    at::Tensor mxscale2;

    TORCH_CHECK(input.dim() >= MIN_INPUT_DIM, "The input should be at least 2D" + OPS_ERROR(ErrCode::PARAM));
    auto y1_shape = op_infer::array_to_small_vector(input.sizes());
    auto mxscale1_shape = op_infer::array_to_small_vector(input.sizes());
    mxscale1_shape.emplace_back(ALIGN_NUM);
    auto y2_shape = op_infer::array_to_small_vector(input.sizes());
    auto mxscale2_shape = op_infer::array_to_small_vector(input.sizes());
    mxscale2_shape.emplace_back(ALIGN_NUM);

    int64_t last_axis = -1;
    int64_t second_to_last_axis = -2;
    int64_t last_axis_change = last_axis + input.dim();
    int64_t second_to_last_axis_change = second_to_last_axis + input.dim();
    int64_t block_size = 32;
    int64_t last_dim_size = op_infer::CeilDiv(mxscale1_shape[last_axis_change], block_size);
    int64_t second_to_dim_size = op_infer::CeilDiv(mxscale2_shape[second_to_last_axis_change], block_size);
    last_dim_size = (last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    second_to_dim_size = (second_to_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    mxscale1_shape[last_axis_change] = last_dim_size;
    mxscale2_shape[second_to_last_axis_change] = second_to_dim_size;
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    aclDataType y_acltype;
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    ASCEND_LOGI("[npu_dynamic_mx_quant_with_dual_axis]: Getting aclTensor y1 and y2 dtype by Parameter(dst_type): %ld", dst_type);
    if (special_output_type) {
        int64_t y1_last_dim_val = y1_shape[input.dim() - 1];
        int64_t y2_last_dim_val = y2_shape[input.dim() - 1];
        TORCH_CHECK(y1_last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "y1 dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(y2_last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "y2 dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
        y1_shape[input.dim() - 1] = y1_last_dim_val / FP4_IN_UINT8_NUM;
        y2_shape[input.dim() - 1] = y1_last_dim_val / FP4_IN_UINT8_NUM;
        y1 = npu_preparation::apply_tensor_without_format(y1_shape, c10::ScalarType::Byte);
        y2 = npu_preparation::apply_tensor_without_format(y2_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y1 = npu_preparation::apply_tensor_without_format(y1_shape, c10::dtype(scalar_dtype));
        y2 = npu_preparation::apply_tensor_without_format(y2_shape, c10::dtype(scalar_dtype));
    }

    mxscale1 = npu_preparation::apply_tensor_without_format(mxscale1_shape, c10::dtype(at::ScalarType::Byte));
    mxscale2 = npu_preparation::apply_tensor_without_format(mxscale2_shape, c10::dtype(at::ScalarType::Byte));
    TensorWrapper y1_wrapper = {y1, y_acltype};
    TensorWrapper y2_wrapper = {y2, y_acltype};
    TensorWrapper mxscale1_wrapper = {mxscale1, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper mxscale2_wrapper = {mxscale2, aclDataType::ACL_FLOAT8_E8M0};
    EXEC_NPU_CMD(aclnnDynamicMxQuantWithDualAxis, input, round_mode_ptr, y_acltype, scale_alg, y1_wrapper, mxscale1_wrapper, y2_wrapper, mxscale2_wrapper);
    
    return std::make_tuple(y1, mxscale1, y2, mxscale2);
}

} // namespace op_api