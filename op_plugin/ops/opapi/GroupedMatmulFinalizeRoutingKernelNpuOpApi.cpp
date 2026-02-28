// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

namespace op_api {
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
using npu_preparation = at_npu::native::OpPreparation;

static bool is_nz_format(const at::Tensor& w)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(w)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

static bool is_weight_trans(const at::Tensor &tensor)
{
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    return tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2);
}

at::Tensor npu_grouped_matmul_finalize_routing(
    const at::Tensor &x,
    const at::Tensor &w,
    const at::Tensor &group_list,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& offset,
    const c10::optional<at::Tensor>& pertoken_scale,
    const c10::optional<at::Tensor>& shared_input,
    const c10::optional<at::Tensor>& logit,
    const c10::optional<at::Tensor>& row_index,
    c10::optional<at::ScalarType> dtype,
    c10::optional<double> shared_input_weight,
    c10::optional<int64_t> shared_input_offset,
    c10::optional<int64_t> output_bs,
    c10::optional<int64_t> group_list_type,
    const c10::OptionalIntArrayRef tuning_config,
    c10::optional<int64_t> x_dtype,
    c10::optional<int64_t> w_dtype,
    c10::optional<int64_t> scale_dtype,
    c10::optional<int64_t> pertoken_scale_dtype
    )
{
    bool is_weight_nz = is_nz_format(w);
    TORCH_CHECK(group_list_type == 1 || group_list_type == 0,
                "only support group_list_type's value 0 or 1.",
                OPS_ERROR(ErrCode::PARAM));

    auto x_dim_num = x.dim();
    auto w_dim_num = w.dim();

    constexpr int EXPECTED_X_DIM = 2;
    constexpr int EXPECTED_W_DIM = 3;

    TORCH_CHECK(x_dim_num == EXPECTED_X_DIM && w_dim_num == EXPECTED_W_DIM,
                "x dim is ", EXPECTED_X_DIM, " weight dim is ", EXPECTED_W_DIM,
                OPS_ERROR(ErrCode::PARAM));

    if (x_dtype.has_value()) {
        TORCH_CHECK(x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                        x_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
            "The optional parameter x_dtype only supports float4_e2m1fn_x2, hifloat8 or None, but now is ",
            c10_npu::CustomDataTypeToString(x_dtype.value()),
            "." + OPS_ERROR(ErrCode::VALUE));
    }

    if (w_dtype.has_value()) {
        TORCH_CHECK(w_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                        w_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
            "The optional parameter w_dtype only supports float4_e2m1fn_x2, hifloat8 or None, but now is ",
            c10_npu::CustomDataTypeToString(w_dtype.value()),
            "." + OPS_ERROR(ErrCode::VALUE));
    }

    if (scale_dtype.has_value()) {
        TORCH_CHECK(scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter scale_dtype only supports float8_e8m0fnu or None, but now is ",
                    c10_npu::CustomDataTypeToString(scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    if (pertoken_scale_dtype.has_value()) {
        TORCH_CHECK(pertoken_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter pertoken_scale_dtype only supports float8_e8m0fnu or None, but now is ",
                    c10_npu::CustomDataTypeToString(pertoken_scale_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    auto x_m_dim = x.size(x_dim_num - LAST_SECOND_DIM_INDEX);
    auto x_k_dim = x.size(x_dim_num - 1);
    auto w_n_dim = w.size(w_dim_num - 1);
    auto w_k_dim = w.size(w_dim_num - LAST_SECOND_DIM_INDEX);

    auto output_size = op_infer::array_to_small_vector(x.sizes());
    int32_t output_bs_real = static_cast<int32_t>(output_bs.value_or(0));
    if (output_bs_real == 0) {
        output_bs_real = x_m_dim;
    }
    output_size[x_dim_num - LAST_SECOND_DIM_INDEX] = output_bs_real;

    static const bool mxfp4_valid = x_dtype.has_value() && w_dtype.has_value() &&
                                    x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) &&
                                    w_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1);
    bool weight_trans = is_weight_trans(w);
    if (w.dtype() == at::kInt) {
        output_size[x_dim_num - 1] = w_n_dim * INT4_NUMS_IN_INT32;
    } else if (mxfp4_valid && !weight_trans) {
        output_size[x_dim_num - 1] = w_n_dim * FP4_IN_INT8;
    } else {
        output_size[x_dim_num - 1] = w_n_dim;
    }

    const at::Tensor &scale_real = scale.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
    const at::Tensor &shared_input_real = shared_input.value_or(at::Tensor());
    const at::Tensor &logit_real = logit.value_or(at::Tensor());
    const at::Tensor &row_index_real = row_index.value_or(at::Tensor());
    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    float shared_input_weight_real = static_cast<float>(shared_input_weight.value_or(1.0));
    int64_t shared_input_offset_real = shared_input_offset.value_or(0);
    int64_t group_list_type_real = group_list_type.value_or(1);
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});
    auto antiquant_scale_real = at::Tensor();
    auto antiquant_offset_real = at::Tensor();

    TensorWrapper x_wrapper = make_wrapper(x, x_dtype);
    TensorWrapper w_wrapper = make_wrapper(w, w_dtype);
    TensorWrapper scale_wrapper = make_wrapper(scale_real, scale_dtype);
    TensorWrapper pertoken_scale_wrapper = make_wrapper(pertoken_scale_real, pertoken_scale_dtype);

    at::ScalarType dst_type = c10::value_or_else(dtype, [] {return at::ScalarType::Float;});
    TORCH_CHECK(dst_type == at::ScalarType::Float,
        "The dtype should be float", OPS_ERROR(ErrCode::PARAM));

    if (shared_input.has_value()) {
        TORCH_CHECK(dst_type == at::ScalarType::Float,
                    "When shared_input and logit is not None, the dtype must be float32",
                    OPS_ERROR(ErrCode::PARAM));
    }
    c10::TensorOptions options = x.options().dtype(dst_type);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    bool transposeX = false;
    bool transposeW = false;
    int64_t dtype_real = 0;
    if (is_weight_nz) {
        static const bool is_v2_available = check_aclnn_kernel_available("aclnnGroupedMatmulFinalizeRoutingWeightNzV2");
        if (is_v2_available) {
            EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingWeightNzV2, x_wrapper, w_wrapper, scale_wrapper, bias_real, offset_real, antiquant_scale_real, antiquant_offset_real, pertoken_scale_wrapper, group_list, shared_input_real, logit_real,
                         row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW, group_list_type_real, tuning_config_real, result);
        } else {
            if (tuning_config.has_value()) {
                TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::npu_grouped_matmul_finalize_routing is "
                    "not support tuning_config, Please try to update your CANN version.");
            }
            EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingWeightNz, x_wrapper, w_wrapper, scale_wrapper, bias_real, pertoken_scale_wrapper, group_list, shared_input_real, logit_real,
                         row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW, group_list_type_real, result);
        }
        return result;
    }

    static const bool is_v3_available = check_aclnn_kernel_available("aclnnGroupedMatmulFinalizeRoutingV3");
    if (is_v3_available) {
        EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingV3, x_wrapper, w_wrapper, scale_wrapper, bias_real, offset_real, antiquant_scale_real, antiquant_offset_real, pertoken_scale_wrapper,
                     group_list, shared_input_real, logit_real, row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW,
                     group_list_type_real, tuning_config_real, result);
    } else {
        if (tuning_config.has_value()) {
            TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::npu_grouped_matmul_finalize_routing is "
                "not support tuning_config, Please try to update your CANN version.");
        }
        EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingV2, x_wrapper, w_wrapper, scale_wrapper, bias_real, offset_real, antiquant_scale_real, antiquant_offset_real, pertoken_scale_wrapper,
                     group_list, shared_input_real, logit_real, row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW,
                     group_list_type_real, result);
    }
    return result;
}

}