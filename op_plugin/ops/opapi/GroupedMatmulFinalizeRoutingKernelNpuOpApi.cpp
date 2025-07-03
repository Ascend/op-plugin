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
    c10::optional<int64_t> group_list_type
    )
{
    bool is_weight_nz = is_nz_format(w);
    TORCH_CHECK(group_list_type == 1,
                "only support group_list_type's value is 1.",
                OPS_ERROR(ErrCode::PARAM));

    auto x_dim_num = x.dim();
    auto w_dim_num = w.dim();

    constexpr int EXPECTED_X_DIM = 2;
    constexpr int EXPECTED_W_DIM = 3;

    TORCH_CHECK(x_dim_num == EXPECTED_X_DIM && w_dim_num == EXPECTED_W_DIM,
                "x dim is ", EXPECTED_X_DIM, " weight dim is ", EXPECTED_W_DIM,
                OPS_ERROR(ErrCode::PARAM));

    auto x_m_dim = x.size(x_dim_num - LAST_SECOND_DIM_INDEX);
    auto x_k_dim = x.size(x_dim_num - 1);
    auto w_n_dim = w.size(w_dim_num - 1);
    auto w_k_dim = w.size(w_dim_num - LAST_SECOND_DIM_INDEX);

    auto output_size = op_infer::array_to_small_vector(x.sizes());
    int32_t output_bs_real = static_cast<int32_t>(output_bs.value_or(0));
    if (!shared_input.has_value() && !logit.has_value()) {
        TORCH_CHECK(output_bs_real == x_m_dim,
                    "When shared_input and logit is None, output_bs must equal to M",
                    OPS_ERROR(ErrCode::PARAM));
    }
    if (output_bs_real == 0) {
        output_bs_real = x_m_dim;
    }
    output_size[x_dim_num - LAST_SECOND_DIM_INDEX] = output_bs_real;
    if (w.dtype() == at::kInt) {
        output_size[x_dim_num - 1] = w_n_dim * INT4_NUMS_IN_INT32;
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
    auto antiquant_scale_real = at::Tensor();
    auto antiquant_offset_real = at::Tensor();

    auto scene_has_share = false;
    auto scene_no_share = false;
    if (scale.has_value() &&  pertoken_scale.has_value() && shared_input.has_value()
        && logit.has_value() && row_index.has_value()) {
        scene_has_share = true;
    }

    if (scale.has_value() &&  pertoken_scale.has_value() && !shared_input.has_value()
        && !logit.has_value() && row_index.has_value()) {
        scene_no_share = true;
    }

    TORCH_CHECK(scene_has_share || scene_no_share,
                "input tensor only support shared_input and logit empty tensor",
                OPS_ERROR(ErrCode::PARAM));
    
    at::ScalarType dst_type = c10::value_or_else(dtype, [] {return at::ScalarType::Float;});
    TORCH_CHECK(dst_type == at::ScalarType::Float,
        "The dtype should be float", OPS_ERROR(ErrCode::PARAM));

    if (shared_input.has_value() && logit.has_value()) {
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
        EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingWeightNz, x, w, scale_real, bias_real, pertoken_scale_real, group_list, shared_input_real, logit_real,
                     row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW, group_list_type_real, result);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulFinalizeRoutingV2, x, w, scale_real, bias_real, offset_real, antiquant_scale_real, antiquant_offset_real, pertoken_scale_real,
                     group_list, shared_input_real, logit_real, row_index_real, dtype_real, shared_input_weight_real, shared_input_offset_real, transposeX, transposeW,
                     group_list_type_real, result);
    }
    return result;
}

}
