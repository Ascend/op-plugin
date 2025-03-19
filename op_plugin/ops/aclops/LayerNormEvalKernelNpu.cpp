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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_layer_norm_eval(const at::Tensor &input, at::IntArrayRef normalized_shape,
                               const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias,
                               double eps)
{
    const at::Tensor &weight_opt = c10::value_or_else(weight, [] { return at::Tensor(); });
    const at::Tensor &bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });
    const int normalized_ndim = static_cast<int>(normalized_shape.size());
    const auto input_shape = input.sizes();
    const auto input_ndim = input.dim();
    const int axis = input_ndim - normalized_ndim;
    const int64_t N = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1LL, std::multiplies<int64_t>());

    at::Tensor result = npu_preparation::apply_tensor(input);
    int64_t numels = 1;
    int64_t begin_dim = 0;
    c10::SmallVector<int64_t, SIZE> tmp_size;
    for (int64_t i = input.dim() - 1; i >= 0; i--) {
        numels *= input.size(i);
        tmp_size.emplace_back(input.size(i));
        if (numels == N) {
            begin_dim = i;
            break;
        }
    }

    std::reverse(tmp_size.begin(), tmp_size.end());
    at::Tensor resize_weight = weight_opt.defined() ? weight_opt.detach().clone() : at::Tensor();
    at::Tensor resize_bias = bias_opt.defined() ? bias_opt.detach().clone() : at::Tensor();
    if (!resize_weight.defined()) {
        resize_weight = at::ones(tmp_size, input.options());
    } else if (!resize_weight.sizes().equals(tmp_size)) {
        resize_weight.resize_(tmp_size);
    }
    if (!resize_bias.defined()) {
        resize_bias = at::zeros(tmp_size, input.options());
    } else if (!resize_bias.sizes().equals(tmp_size)) {
        resize_bias.resize_(tmp_size);
    }

    c10::SmallVector<int64_t, SIZE> output_size;
    for (int64_t i = 0; i < input_ndim; i++) {
        if (i < begin_dim) {
            output_size.emplace_back(input.size(i));
        } else {
            output_size.emplace_back(1);
        }
    }

    at::Tensor mean = npu_preparation::apply_tensor(resize_weight, output_size);
    at::Tensor rstd = npu_preparation::apply_tensor(resize_weight, output_size);
    at_npu::native::OpCommand cmd;
    cmd.Name("LayerNormV3")
        .Input(input)
        .Input(resize_weight)
        .Input(resize_bias)
        .Output(result)
        .Output(mean)
        .Output(rstd)
        .Attr("begin_norm_axis", begin_dim)
        .Attr("begin_params_axis", begin_dim)
        .Attr("epsilon", static_cast<float>(eps))
        .Run();
    return result;
}
} // namespace acl_op
