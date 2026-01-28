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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t ROUND_MODE_RINT = 0;
const int64_t ROUND_MODE_TRUNC = 4;
const int64_t SWISH_NUM = 2;

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale, const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index, bool activate_left, int64_t quant_mode,
    c10::optional<int64_t> dst_type, c10::optional<int64_t> round_mode, c10::optional<int64_t> activate_dim,
    int64_t swiglu_mode, double clamp_limit, double glu_alpha, double glu_bias)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got", quant_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0.0, "clamp_limit should be positive finite",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(std::isfinite(glu_bias),  "glu_bias should be finite",  OPS_ERROR(ErrCode::PARAM));

    static const bool is_v2_available = check_aclnn_kernel_available("aclnnDequantSwigluQuantV2");

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;

    std::string quant_mode_str = "static";
    if (quant_mode == 1) {
        quant_mode_str = "dynamic";
    }
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
    const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    const at::Tensor& weight_scale_opt = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
    const at::Tensor& activate_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
    const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });

    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(x.size(x.dim() - 1) % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));

        for (int i = 0; i < x.dim() - 1; i++) {
            y_size.push_back(x.size(i));
            scale_size.push_back(x.size(i));
        }
        auto last_dim = x.size(x.dim() - 1) / SWISH_NUM;
        y_size.push_back(last_dim);

        at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
        at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

        if (swiglu_mode == 0 && !is_v2_available) {
            EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                         quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);
        } else {
            int64_t dst_type = 2;
            char* round_mode = "rint";
            int64_t activate_dim = -1;
            EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                         quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, dst_type, round_mode,
                         activate_dim, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale);
        }

        return std::tie(y, scale);
    } else {
        bool is_v1_available = !dst_type.has_value() && !round_mode.has_value() && !activate_dim.has_value();
        int64_t dst_type_value = dst_type.has_value() ? dst_type.value() : static_cast<int>(at::ScalarType::Char);
        int64_t round_mode_value = round_mode.has_value() ? round_mode.value() : 0;
        int64_t activate_dim_value = activate_dim.has_value() ? activate_dim.value() : -1;

        if (!is_v2_available || is_v1_available) {
            TORCH_CHECK(x.size(x.dim() - 1) % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));

            for (int i = 0; i < x.dim() - 1; i++) {
                y_size.push_back(x.size(i));
                scale_size.push_back(x.size(i));
            }
            auto last_dim = x.size(x.dim() - 1) / SWISH_NUM;
            y_size.push_back(last_dim);

            at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
            at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

            EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                         quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);

            return std::tie(y, scale);
        } else {
            TORCH_CHECK(round_mode_value >= ROUND_MODE_RINT && round_mode_value <= ROUND_MODE_TRUNC,
                        "round_mode only support [0, 1, 2, 3, 4], but got", round_mode_value, OPS_ERROR(ErrCode::PARAM));
    
            // transform activate_dim
            if (activate_dim_value < 0) {
                activate_dim_value = activate_dim_value + x.dim();
            }
            TORCH_CHECK(activate_dim_value <= (x.dim() - 1) && activate_dim_value >= 0, "activate_dim should less than ", x.dim() - 1, OPS_ERROR(ErrCode::PARAM));

            TORCH_CHECK(x.size(activate_dim_value) % SWISH_NUM == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));

            for (int i = 0; i < x.dim(); i++) {
                if (i == activate_dim_value) {
                    y_size.push_back(x.size(i) / SWISH_NUM);
                } else {
                    y_size.push_back(x.size(i));
                }
            }

            bool special_output_type = (dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                        dst_type_value == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));

            if (special_output_type) {
                int64_t last_dim_val = y_size[x.dim() - 1];
                y_size[x.dim() - 1] = last_dim_val / SWISH_NUM;
                TORCH_CHECK(last_dim_val % SWISH_NUM == 0, "Y last dim should be even when the type of y is float4_e1m2 or float4_e2m1", OPS_ERROR(ErrCode::PARAM));
            }

            for (int i = 0; i < x.dim() - 1; i++) {
                if (i == activate_dim_value) {
                    scale_size.push_back(x.size(i) / SWISH_NUM);
                } else {
                    scale_size.push_back(x.size(i));
                }
            }

            at::Tensor y;
            aclDataType y_acltype;
        
            if (special_output_type) {
                y = npu_preparation::apply_tensor_without_format(y_size, c10::ScalarType::Byte);
                y_acltype = c10_npu::GetAclDataType(dst_type_value);
            } else {
                y_acltype = c10_npu::GetAclDataType(dst_type_value);
                at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
                y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(scalar_dtype));
            }

            TensorWrapper y_wrapper = {y, y_acltype};

            at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

            static const char* round_modes[] = {"rint", "round", "floor", "ceil", "trunc"};
            const char* round_mode_ptr = round_modes[round_mode_value];

            EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                         quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y_acltype, round_mode_ptr,
                         activate_dim_value, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y_wrapper, scale);

            return std::tie(y, scale);
        }
    }
}
}  // namespace op_api
