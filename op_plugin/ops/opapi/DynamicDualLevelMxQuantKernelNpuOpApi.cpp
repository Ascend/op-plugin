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
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_dual_level_mx_quant(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scale,
    c10::string_view round_mode)
{
    at::Tensor y;
    at::Tensor level0_scale;
    at::Tensor level1_scale;
    auto y_shape = op_infer::array_to_small_vector(input.sizes());
    auto level0_scale_shape = op_infer::array_to_small_vector(input.sizes());
    auto level1_scale_shape = op_infer::array_to_small_vector(input.sizes());
    level1_scale_shape.emplace_back(ALIGN_NUM);

    int64_t level0_block_size = 512;
    int64_t level1_block_size = 32;
    int64_t dim0_size = op_infer::CeilDiv(level0_scale_shape[input.dim() - 1], level0_block_size);
    level0_scale_shape[input.dim() - 1] = dim0_size;
    int64_t dim1_size = op_infer::CeilDiv(level1_scale_shape[input.dim() - 1], level1_block_size);
    dim1_size = (dim1_size + ALIGN_NUM - 1) / ALIGN_NUM;
    level1_scale_shape[input.dim() - 1] = dim1_size;
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // prepare for empty output tensor
    aclDataType y_acltype = aclDataType::ACL_FLOAT4_E2M1;
    int64_t last_dim_val = y_shape[input.dim() - 1];
    TORCH_CHECK(last_dim_val % FP4_IN_UINT8_NUM == 0,
                "The last dim input shape must be divisible by 2 if "
                "output dtype is torch_npu.float4_e2m1" + OPS_ERROR(ErrCode::PARAM));
    y_shape[input.dim() - 1] = last_dim_val / FP4_IN_UINT8_NUM;
    y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);

    level0_scale = npu_preparation::apply_tensor_without_format(level0_scale_shape, c10::dtype(at::ScalarType::Float));
    level1_scale = npu_preparation::apply_tensor_without_format(level1_scale_shape, c10::dtype(at::ScalarType::Byte));
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper level1_scale_wrapper = {level1_scale, aclDataType::ACL_FLOAT8_E8M0};
    EXEC_NPU_CMD(aclnnDynamicDualLevelMxQuant, input, smooth_scale, round_mode_ptr, level0_block_size, level1_block_size, y_wrapper, level0_scale, level1_scale_wrapper);
    
    return std::make_tuple(y, level0_scale, level1_scale);
}

} // namespace op_api