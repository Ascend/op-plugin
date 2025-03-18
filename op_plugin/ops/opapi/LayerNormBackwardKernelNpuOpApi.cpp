// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

tensor_list3 native_layer_norm_backward(const at::Tensor &grad_out, const at::Tensor &input, at::IntArrayRef normalized_shape,
                                        const at::Tensor &mean, const at::Tensor &rstd,
                                        const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias,
                                        std::array<bool, 3> output_mask)
{
    DO_COMPATIBILITY(aclnnLayerNormBackward, acl_op::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd,
        weight, bias, output_mask));
    const at::Tensor &weight_value = c10::value_or_else(weight, [] { return at::Tensor(); });
    const at::Tensor &bias_value = c10::value_or_else(bias, [] { return at::Tensor(); });
    at::Tensor weight_refined =
        weight_value.defined() ? weight_value.resize_(normalized_shape) : at::ones(normalized_shape, input.options());
    at::Tensor bias_refined =
        bias_value.defined() ? bias_value.resize_(normalized_shape) : at::zeros(normalized_shape, input.options());

    // 根据输入input和normalized_shape计算M
    const size_t norm_dim = normalized_shape.size();
    const auto input_shape = input.sizes();
    const size_t input_dim = static_cast<size_t>(input.dim());
    const size_t begin_axis = input_dim - norm_dim;

    const int64_t M =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());

    at::SmallVector<int64_t, SIZE> mean_shape = op_infer::array_to_small_vector(input.sizes());
    for (size_t index = begin_axis; index < input_dim; index++) {
        mean_shape[index] = 1;
    }
    at::Tensor mean_refined = mean.reshape(mean_shape);
    at::Tensor variance_refined = rstd.reshape(mean_shape);

    // 构造输出tensor
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    // 根据mask初始化输出tensor
    if (output_mask[0]) {
        grad_input =
            at::native::empty_like(input, c10::nullopt /* dtype */, c10::nullopt /* layout */, c10::nullopt /* device */,
                                   c10::nullopt /* pin_memory */, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
        grad_weight = at::native::zeros_like(weight_refined, at::kFloat /* dtype */, c10::nullopt /* layout */,
                                             c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                             LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[2]) {
        grad_bias = at::native::zeros_like(bias_refined, at::kFloat /* dtype */, c10::nullopt /* layout */,
                                           c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                           LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    // 调用HostAPI接口
    EXEC_NPU_CMD(aclnnLayerNormBackward, grad_out, input, normalized_shape, mean_refined, variance_refined, weight_refined,
                 bias_refined, output_mask, grad_input, grad_weight, grad_bias);
    return std::tie(grad_input, grad_weight, grad_bias);
}

}
