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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_grouped_matmul_swiglu_quant(
    const at::Tensor & x,
    const at::Tensor & weight,
    const at::Tensor & group_list,
    const at::Tensor & weight_scale,
    const at::Tensor & x_scale,
    const c10::optional<at::Tensor> & bias,
    const c10::optional<at::Tensor> & offset)
{
    auto x_size = x.sizes();
    int n = weight.sizes()[2];
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = npu_preparation::apply_tensor_without_format({m, n/2}, c10::dtype(c10::ScalarType::Char));
    at::Tensor output_scale = npu_preparation::apply_tensor_without_format(x_scale, {m});
    at::Tensor output_offset = npu_preparation::apply_tensor_without_format({}, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNZ,
        x,
        weight,
        bias,
        offset,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale,
        output_offset);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}
}
