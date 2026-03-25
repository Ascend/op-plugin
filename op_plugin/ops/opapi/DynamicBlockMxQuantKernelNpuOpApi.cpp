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
constexpr int64_t BLOCK_SIZE_BASE_NUM = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t MIN_INPUT_DIM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_block_mx_quant(
    const at::Tensor &input,
    c10::string_view round_mode,
    int64_t dst_type,
    int64_t scale_alg,
    double dst_type_max)
{
    at::Tensor y;
    at::Tensor scale1;
    at::Tensor scale2;

    TORCH_CHECK(input.dim() >= MIN_INPUT_DIM, "The input should be at least 2D" + OPS_ERROR(ErrCode::PARAM));
    auto y_shape = op_infer::array_to_small_vector(input.sizes());
    auto scale1_shape = op_infer::array_to_small_vector(input.sizes());
    auto scale2_shape = op_infer::array_to_small_vector(input.sizes());
    scale1_shape.emplace_back(ALIGN_NUM);
    scale2_shape.emplace_back(ALIGN_NUM);

    int64_t last_axis = -1;
    int64_t second_to_last_axis = -2;
    int64_t last_axis_change = last_axis + input.dim();
    int64_t second_to_last_axis_change = second_to_last_axis + input.dim();
    int64_t block_size = 32;
    int64_t last_dim_size = op_infer::CeilDiv(scale1_shape[last_axis_change], block_size);
    int64_t second_to_dim_size = op_infer::CeilDiv(scale2_shape[second_to_last_axis_change], block_size);
    last_dim_size = (last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    second_to_dim_size = (second_to_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    scale1_shape[last_axis_change] = last_dim_size;
    scale2_shape[second_to_last_axis_change] = second_to_dim_size;
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    aclDataType y_acltype;
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    ASCEND_LOGI("[npu_dynamic_block_mx_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);
    if (special_output_type) {
        int64_t y_last_dim_val = y_shape[input.dim() - 1];
        TORCH_CHECK(y_last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "y dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
        y_shape[input.dim() - 1] = y_last_dim_val / FP4_IN_UINT8_NUM;
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }

    scale1 = npu_preparation::apply_tensor_without_format(scale1_shape, c10::dtype(at::ScalarType::Byte));
    scale2 = npu_preparation::apply_tensor_without_format(scale2_shape, c10::dtype(at::ScalarType::Byte));
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper scale1_wrapper = {scale1, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper scale2_wrapper = {scale2, aclDataType::ACL_FLOAT8_E8M0};
    EXEC_NPU_CMD(aclnnDynamicBlockMxQuant, input, round_mode_ptr, y_acltype, scale_alg, dst_type_max, y_wrapper, scale1_wrapper, scale2_wrapper);
    
    return std::make_tuple(y, scale1, scale2);
}

} // namespace op_api