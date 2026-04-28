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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
constexpr int64_t NUM_TWO = 2;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
constexpr int64_t MIN_INPUT_DIM = 2;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_swiglu_mx_quant_with_dual_axis(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    c10::string_view round_mode,
    int64_t scale_alg,
    int64_t dst_type,
    double dst_type_max)
{
    at::Tensor y1;
    at::Tensor mxscale1;
    at::Tensor y2;
    at::Tensor mxscale2;

    TORCH_CHECK(x.dim() >= MIN_INPUT_DIM, "The input x should be at least 2D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.size(-1) % NUM_TWO == 0, "The last dim of input must be divisible by 2" + OPS_ERROR(ErrCode::PARAM));

    static const bool is_available = check_aclnn_kernel_available("aclnnSwigluMxQuantWithDualAxis");
    TORCH_CHECK(is_available,
                "Current CANN version do not support this api: npu_swiglu_mx_quant_with_dual_axis. Please try to update the version of CANN."
                + OPS_ERROR(ErrCode::PARAM));

    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // Infer y shape: divide last dim by 2
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    y_shape[y_shape.size() - 1] = y_shape[y_shape.size() - 1] / NUM_TWO;

    // Infer mxscale1 shape: ceil(last_dim / 64) + append 2
    auto mxscale1_shape = op_infer::array_to_small_vector(y_shape);
    int64_t last_dim = mxscale1_shape[mxscale1_shape.size() - 1];
    mxscale1_shape[mxscale1_shape.size() - 1] = static_cast<int64_t>(std::ceil(static_cast<double>(last_dim) / SPLIT_BLOCK_SIZE));
    mxscale1_shape.emplace_back(NUM_TWO);

    // Infer mxscale2 shape: floor(second_to_last_dim / 64) + group_num + append 2
    auto mxscale2_shape = op_infer::array_to_small_vector(y_shape);
    int64_t second_to_last_dim = mxscale2_shape[mxscale2_shape.size() - 2];
    int64_t quant_size = static_cast<int64_t>(std::floor(static_cast<double>(second_to_last_dim) / SPLIT_BLOCK_SIZE));
    if (group_index_opt.defined()) {
        quant_size = quant_size + group_index_opt.size(0);
    }
    mxscale2_shape[mxscale2_shape.size() - 2] = quant_size;
    mxscale2_shape.emplace_back(NUM_TWO);

    aclDataType y_acltype;
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));

    ASCEND_LOGI("[npu_swiglu_mx_quant_with_dual_axis]: Getting aclTensor y1 and y2 dtype by Parameter(dst_type): %ld", dst_type);

    if (special_output_type) {
        int64_t y_last_dim_val = y_shape[y_shape.size() - 1];
        TORCH_CHECK(y_last_dim_val % NUM_TWO == 0,
                    "The last dim of y must be divisible by 2 if y dtype is float4_e2m1 or float4_e1m2"
                    + OPS_ERROR(ErrCode::PARAM));
        y_shape[y_shape.size() - 1] = y_last_dim_val / NUM_TWO;
        y1 = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y2 = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y1 = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
        y2 = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }

    mxscale1 = npu_preparation::apply_tensor_without_format(mxscale1_shape, c10::dtype(at::ScalarType::Byte));
    mxscale2 = npu_preparation::apply_tensor_without_format(mxscale2_shape, c10::dtype(at::ScalarType::Byte));

    TensorWrapper y1_wrapper = {y1, y_acltype};
    TensorWrapper y2_wrapper = {y2, y_acltype};
    TensorWrapper mxscale1_wrapper = {mxscale1, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper mxscale2_wrapper = {mxscale2, aclDataType::ACL_FLOAT8_E8M0};

    EXEC_NPU_CMD(aclnnSwigluMxQuantWithDualAxis, x, group_index_opt, activate_left,
                 round_mode_ptr, scale_alg, y_acltype, dst_type_max,
                 y1_wrapper, mxscale1_wrapper, y2_wrapper, mxscale2_wrapper);

    return std::make_tuple(y1, mxscale1, y2, mxscale2);
}

} // namespace op_api
