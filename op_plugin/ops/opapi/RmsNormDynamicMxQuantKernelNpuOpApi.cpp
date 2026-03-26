// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;
constexpr int64_t MIN_INPUT_DIM = 1;
constexpr int64_t MAX_INPUT_DIM = 7;
}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rms_norm_dynamic_mx_quant(
    const at::Tensor &x,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon,
    const int64_t scale_alg,
    c10::string_view round_mode,
    int64_t dst_type)
{
    // output
    at::Tensor y;
    at::Tensor mxscale;
    at::Tensor rstd;

    // check params
    TORCH_CHECK(x.dim() >= MIN_INPUT_DIM && x.dim() <= MAX_INPUT_DIM, "The x should be in 1~7D" + OPS_ERROR(ErrCode::PARAM));

    static const bool is_available = check_aclnn_kernel_available("aclnnRmsNormDynamicMxQuant");
    TORCH_CHECK(is_available,
                "Current CANN version do not support this api. Please try to update the version of CANN."
                + OPS_ERROR(ErrCode::PARAM));

    // y
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    aclDataType y_acltype = c10_npu::GetAclDataType(dst_type);
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    ASCEND_LOGI("[npu_rms_norm_dynamic_mx_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);
    if (special_output_type) {
        int64_t last_dim_val = y_shape[x.dim() - 1];
        TORCH_CHECK(last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim x shape must be divisible by 2 if "
                    "output dtype is torch_npu.float4_e2m1 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
        y_shape[x.dim() - 1] = last_dim_val / FP4_IN_UINT8_NUM;
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
    } else {
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }

    // mxscale
    auto mxscale_shape = op_infer::array_to_small_vector(x.sizes());
    mxscale_shape.emplace_back(ALIGN_NUM);
    int64_t last_axis_change = x.dim() - 1;
    int64_t dim_size = op_infer::CeilDiv(mxscale_shape[last_axis_change], BLOCK_SIZE);
    dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    mxscale_shape[last_axis_change] = dim_size;
    mxscale = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));

    // rstd
    bool output_rstd = x.requires_grad();
    if (output_rstd) {
        auto output_size = op_infer::rms_norm_npu_output_size(x, gamma);
        rstd = npu_preparation::apply_tensor_with_format(output_size[1], x.options().dtype(at::kFloat), ACL_FORMAT_ND);
    } else {
        rstd = at::empty({0}, x.options().dtype(at::kFloat));
    }

    // call aclnn
    char *round_mode_ptr = const_cast<char *>(round_mode.data());
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale, aclDataType::ACL_FLOAT8_E8M0};

    EXEC_NPU_CMD(aclnnRmsNormDynamicMxQuant, x, gamma, beta, epsilon, scale_alg,
                 round_mode_ptr, y_acltype, output_rstd, y_wrapper, mxscale_wrapper, rstd);
    
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, mxscale, rstd);
}

} // namespace op_api