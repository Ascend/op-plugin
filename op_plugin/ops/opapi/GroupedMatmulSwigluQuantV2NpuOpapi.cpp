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
    const c10::OptionalIntArrayRef tuning_config)
{
    auto x_size = x.sizes();
    int n = weight[0].sizes()[2];
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = npu_preparation::apply_tensor_without_format({m, n/2}, c10::dtype(c10::ScalarType::Char));
    at::Tensor output_scale = npu_preparation::apply_tensor_without_format(x_scale, {m});

    int64_t dequant_mode_real = dequant_mode.value_or(0);
    int64_t dequant_dtype_real = dequant_dtype.value_or(0);
    int64_t quant_mode_real = quant_mode.value_or(0);
    int64_t group_list_type_real = group_list_type.value_or(0);
    auto weight_assist_matrix_real = weight_assist_matrix.value_or(at::TensorList());
    auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});
    auto bias_real = bias.value_or(at::Tensor());
    auto smooth_scale_real = smooth_scale.value_or(at::Tensor());

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNzV2,
        x,
        weight,
        weight_scale,
        weight_assist_matrix_real,
        bias_real,
        x_scale,
        smooth_scale_real,
        group_list,
        dequant_mode_real,
        dequant_dtype_real,
        quant_mode_real,
        group_list_type_real,
        tuning_config_real,
        output,
        output_scale);
    return std::tuple<at::Tensor, at::Tensor>(output, output_scale);
}
}
