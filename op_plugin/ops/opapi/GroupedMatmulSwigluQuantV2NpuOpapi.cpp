// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
constexpr int64_t MXFP_DIVISOR_SIZE = 64LL;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2LL;
constexpr int64_t NUM_TWO = 2LL;
constexpr int64_t NUM_ONE = 1LL;
constexpr int64_t DIM_2 = 2LL;
constexpr int64_t DIM_1 = 1LL;
constexpr int64_t DIM_0 = 0LL;
constexpr int64_t WEIGHT_MAX_DIM_NUM = 3LL;
constexpr int64_t WEIGHT_PENULTIMATE_DIM = 1LL;
constexpr int64_t WEIGHT_LAST_DIM = 2LL;
constexpr int64_t DIM_3 = 3LL;
constexpr int64_t FLOAT8_E5M2 = 35LL;
constexpr int64_t FLOAT8_E4M3FN = 36LL;

void create_new_tensor(at::Tensor &y, size_t dim_m, size_t dim_n, c10::TensorOptions options)
{
    auto output_size = op_infer::array_to_small_vector({dim_m, dim_n});
    y = npu_preparation::apply_tensor_without_format(output_size, options);
}

void create_new_tensor_batch(at::Tensor &y, size_t batch, size_t dim_m, size_t dim_n,
                             const c10::TensorOptions &options)
{
    auto output_size = op_infer::array_to_small_vector({batch, dim_m, dim_n});
    y = npu_preparation::apply_tensor_without_format(output_size, options);
}

std::tuple<at::Tensor, at::Tensor> npu_grouped_matmul_swiglu_quant_v2(
    const at::Tensor & x,
    const at::TensorList weight,
    const at::TensorList weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & smooth_scale,
    const c10::optional<at::TensorList> weight_assist_matrix,
    const c10::optional<at::Tensor> & bias,
    c10::optional<int64_t> dequant_mode,
    c10::optional<int64_t> dequant_dtype,
    c10::optional<int64_t> quant_mode,
    c10::optional<int64_t> quant_dtype,
    c10::optional<int64_t> group_list_type,
    const c10::OptionalIntArrayRef tuning_config,
    c10::optional<int64_t> x_dtype,
    c10::optional<int64_t> weight_dtype,
    c10::optional<int64_t> weight_scale_dtype,
    c10::optional<int64_t> x_scale_dtype)
{
    TORCH_CHECK(weight.size() == 1, "The size of weight should be 1, current size is ", weight.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_scale.size() == 1, "The size of weight_scale should be 1, current size is ",
                weight_scale.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.dim() >= DIM_2, "The x dim should greater than 2, but the actual value is ", x.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!weight_scale[DIM_0].sizes().empty(), "The weight_scale[0] is empty.", OPS_ERROR(ErrCode::PARAM));

    auto x_size = x.sizes();
    int n = weight[DIM_0].sizes()[DIM_2];
    int m = x_size[DIM_0];
    int k = x_size[DIM_1];

    if (x_dtype.has_value()) {
        TORCH_CHECK(x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)
                 || x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1),
                    "The optional parameter x_dtype only supports torch_npu.float4_e2m1fn_x2/torch_npu.float4_e1m2fn_x2 or None, but the actual value is ",
                    c10_npu::CustomDataTypeToString(x_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (weight_dtype.has_value()) {
        TORCH_CHECK(weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)
                 || weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1),
                    "The optional parameter weight_dtype only supports torch_npu.float4_e2m1fn_x2/torch_npu.float4_e1m2fn_x2 or None, but the actual value is ",
                    c10_npu::CustomDataTypeToString(weight_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    TORCH_CHECK((x_dtype.has_value() && weight_dtype.has_value()) || (!x_dtype.has_value() && !weight_dtype.has_value()),
                "The optional parameter x_dtype and weight_dtype should both be torch_npu.float4_e2m1fn_x2/torch_npu.float4_e1m2fn_x2 or None.", OPS_ERROR(ErrCode::VALUE));
    if (weight_scale_dtype.has_value()) {
        TORCH_CHECK(weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter weight_scale_dtype only supports float8_e8m0fnu or None, but the actual value is ",
                    c10_npu::CustomDataTypeToString(weight_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (x_scale_dtype.has_value()) {
        TORCH_CHECK(x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter x_scale_dtype only supports float8_e8m0fnu or None, but the actual value is ",
                    c10_npu::CustomDataTypeToString(x_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    if (dequant_dtype.has_value()) {
        TORCH_CHECK(dequant_dtype.value() == static_cast<int64_t>(c10::ScalarType::Float)
                    || dequant_dtype.value() == static_cast<int64_t>(c10::ScalarType::Char),
                    "The optional parameter dequant_dtype only support torch.float32 or torch.int8, but the actual value is ",
                    c10_npu::CustomDataTypeToString(dequant_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    
    int64_t dequant_mode_real = dequant_mode.value_or(0);
    int64_t dequant_dtype_real = dequant_dtype.value_or(0);
    int64_t quant_mode_real = quant_mode.value_or(0);
    int64_t group_list_type_real = group_list_type.value_or(0);
    auto weight_assist_matrix_real = weight_assist_matrix.value_or(at::TensorList());
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});
    auto bias_real = bias.value_or(at::Tensor());
    auto smooth_scale_real = smooth_scale.value_or(at::Tensor());

    // infer weight is trans or not, when wight is trans, weight_strides[-2] == 1 and weight_strides[-1] == k
    c10::SmallVector<int64_t, WEIGHT_MAX_DIM_NUM> weight_strides = op_infer::array_to_small_vector(weight[DIM_0].strides());
    bool weight_trans = (weight_strides[WEIGHT_PENULTIMATE_DIM] == NUM_ONE && weight_strides[WEIGHT_LAST_DIM] == k);
    static const bool mxfp4_input = x_dtype.has_value() && weight_dtype.has_value() &&
                                   (x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
                                    x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)) &&
                                   (weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
                                    weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1));
    at::Tensor output;
    at::Tensor output_scale;
    if (!weight_scale_dtype.has_value()) {
        output = npu_preparation::apply_tensor_without_format({m, n / MXFP_MULTI_BASE_SIZE}, c10::dtype(c10::ScalarType::Char));
        output_scale = npu_preparation::apply_tensor_without_format({m}, c10::dtype(c10::ScalarType::Float));
    } else {
        if (dequant_dtype.has_value()) {
                dequant_dtype_real = static_cast<int64_t>(c10_npu::GetAclDataType(dequant_dtype.value()));
        }
        TORCH_CHECK(!weight[DIM_0].sizes().empty(), "weight[0] is empty.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(weight[DIM_0].dim() == DIM_3, "weight[0] dim should be equal to 3, but the actual value is ",
                    weight[DIM_0].dim(), OPS_ERROR(ErrCode::PARAM));
        n = weight[DIM_0].sizes()[DIM_2]; // In mx quant mode, n needs to be obtained from the dim 2 of weight.
        c10::TensorOptions options_output = x.options().dtype(quant_dtype.has_value()
                    ? npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(quant_dtype.value()))
                    : x[DIM_0].scalar_type());
        c10::TensorOptions options = x.options().dtype(npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(weight_scale_dtype.value())));

        if (mxfp4_input) {
            if (!weight_trans) {
                if (c10_npu::GetAclDataType(quant_dtype.value()) == FLOAT8_E5M2 || c10_npu::GetAclDataType(quant_dtype.value()) == FLOAT8_E4M3FN) {
                    create_new_tensor(output, m, ((n / MXFP_MULTI_BASE_SIZE) * FP4_IN_INT8), options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n * FP4_IN_INT8 / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE),
                                            MXFP_MULTI_BASE_SIZE, options);
                } else {
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n * FP4_IN_INT8 / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                }
            } else {
                if (c10_npu::GetAclDataType(quant_dtype.value()) == FLOAT8_E5M2 || c10_npu::GetAclDataType(quant_dtype.value()) == FLOAT8_E4M3FN) {
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                } else {
                    create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE / NUM_TWO, options_output);
                    create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
                }
            }
        } else {
            create_new_tensor(output, m, n / MXFP_MULTI_BASE_SIZE, options_output);
            create_new_tensor_batch(output_scale, m, op_infer::CeilDiv(n / MXFP_MULTI_BASE_SIZE, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE, options);
        }
    }

    TensorWrapper x_wrapper = {x,
        x_dtype.has_value() ? c10_npu::GetAclDataType(x_dtype.value())
                            : npu_preparation::convert_to_acl_data_type(x.scalar_type())};
    TensorListWrapper weight_wrapper = {weight,
        weight_dtype.has_value() ? c10_npu::GetAclDataType(weight_dtype.value())
                                 : npu_preparation::convert_to_acl_data_type(weight[0].scalar_type())};
    TensorListWrapper weight_scale_wrapper = {weight_scale,
        weight_scale_dtype.has_value() ? c10_npu::GetAclDataType(weight_scale_dtype.value())
                                : (weight_scale.empty() ? aclDataType::ACL_FLOAT
                                : npu_preparation::convert_to_acl_data_type(weight_scale[0].scalar_type()))};
    TensorWrapper x_scale_wrapper = {x_scale,
        x_scale_dtype.has_value() ? c10_npu::GetAclDataType(x_scale_dtype.value())
                                : (!x_scale.numel() ? aclDataType::ACL_FLOAT
                                : npu_preparation::convert_to_acl_data_type(x_scale.scalar_type()))};
    TensorWrapper output_wrapper = {output,
        quant_dtype.has_value() ? c10_npu::GetAclDataType(quant_dtype.value()): aclDataType::ACL_FLOAT};
    TensorWrapper output_scale_wrapper = {output_scale,
        weight_scale_dtype.has_value() ? aclDataType::ACL_FLOAT8_E8M0 : aclDataType::ACL_FLOAT};

    const bool is_weight_nz = at_npu::native::custom_ops::get_npu_format(weight[DIM_0]) == ACL_FORMAT_FRACTAL_NZ;
    if (is_weight_nz) {
        static const bool is_weight_nz_available = check_aclnn_kernel_available("aclnnGroupedMatmulSwigluQuantWeightNzV2");
        TORCH_CHECK(is_weight_nz_available,
                    "Format of weight in npu_grouped_matmul is FRACTAL_NZ, current CANN version "
                    "do not support with this format. Please try to update the version of CANN."
                    + OPS_ERROR(ErrCode::PARAM));
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantWeightNzV2,
            x,
            weight,
            weight_scale_wrapper,
            weight_assist_matrix_real,
            bias_real,
            x_scale_wrapper,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type_real,
            tuning_config_real,
            output,
            output_scale_wrapper);
    } else {
        EXEC_NPU_CMD(
            aclnnGroupedMatmulSwigluQuantV2,
            x_wrapper,
            weight_wrapper,
            weight_scale_wrapper,
            weight_assist_matrix_real,
            bias_real,
            x_scale_wrapper,
            smooth_scale_real,
            group_list,
            dequant_mode_real,
            dequant_dtype_real,
            quant_mode_real,
            group_list_type_real,
            tuning_config_real,
            output_wrapper,
            output_scale_wrapper);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}
}
