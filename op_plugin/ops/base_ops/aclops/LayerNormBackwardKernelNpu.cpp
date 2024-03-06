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
using tensor_list = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>;
using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
tensor_list layer_norm_backward_nocheck(at::Tensor &d_x, at::Tensor &dgamma, at::Tensor &dbeta, const at::Tensor &d_y,
                                        const at::Tensor &X, const at::Tensor &mean, const at::Tensor &variance,
                                        const at::Tensor &gamma, int64_t M, int64_t N)
{
    at::SmallVector<int64_t, SIZE> tmp_size = op_infer::array_to_small_vector(X.sizes());
    for (int i = X.dim() - gamma.dim(); i < X.dim(); i++) {
        tmp_size[i] = 1;
    }
    at::Tensor mean_ex = mean.reshape(tmp_size);
    at::Tensor variance_ex = variance.reshape(tmp_size);
    double eps = 1e-05;

    at_npu::native::OpCommand cmd;
    cmd.Name("LayerNormGradV3")
        .Input(d_y)
        .Input(X)
        .Input(variance_ex)
        .Input(mean_ex)
        .Input(gamma)
        .Output(d_x)
        .Output(dgamma)
        .Output(dbeta)
        .Run();
    return std::tuple<at::Tensor &, at::Tensor &, at::Tensor &>(d_x, dgamma, dbeta);
}

tensor_list3 layer_norm_backward_npu_support(const at::Tensor &d_y, const at::Tensor &X, const at::Tensor &mean,
                                             const at::Tensor &variance, const c10::optional<at::Tensor> &gamma_ex,
                                             int64_t M, int64_t N, std::array<bool, 3> output_mask)
{
    const at::Tensor &gamma = c10::value_or_else(gamma_ex, [] { return at::Tensor(); });
    at::Tensor d_x;
    at::Tensor dgamma;
    at::Tensor dbeta;
    at::Tensor gamma_temp = gamma;

    at::SmallVector<int64_t, 8> tmp_size;
    int64_t numels = 1;
    for (int64_t i = X.dim() - 1; i >= 0; i--) {
        numels *= X.size(i);
        tmp_size.emplace_back(X.size(i));
        if (numels == N) {
            break;
        }
    }
    std::reverse(tmp_size.begin(), tmp_size.end());
    if (!gamma.defined()) {
        gamma_temp = at::ones(tmp_size, X.options());
    } else if (!gamma.sizes().equals(tmp_size)) {
        gamma_temp.resize_(tmp_size);
    }

    auto output_sizes = op_infer::layer_norm_backward_npu_output_size(d_y, X, mean, variance, gamma_temp, M, N);

    if (M <= 0) {
        d_x = at::native::empty_like(X, c10::nullopt /* dtype */, c10::nullopt /* layout */, c10::nullopt /* device */,
                                     c10::nullopt /* pin_memory */, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        dgamma = at::native::zeros_like(gamma_temp, c10::nullopt /* dtype */, c10::nullopt /* layout */,
                                        c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        dbeta = at::native::zeros_like(gamma_temp, c10::nullopt /* dtype */, c10::nullopt /* layout */,
                                       c10::nullopt /* device */, c10::nullopt /* pin_memory */,
                                       LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        return std::make_tuple(std::move(d_x), std::move(dgamma), std::move(dbeta));
    }

    d_x = npu_preparation::apply_tensor(X, std::get<0>(output_sizes));
    dgamma = npu_preparation::apply_tensor(gamma_temp, std::get<1>(output_sizes));
    dbeta = npu_preparation::apply_tensor(gamma_temp, std::get<2>(output_sizes));

    return layer_norm_backward_nocheck(d_x, dgamma, dbeta, d_y, X, mean, variance, gamma_temp, M, N);
}
} // namespace

tensor_list3 native_layer_norm_backward(const at::Tensor &d_y, const at::Tensor &X, at::IntArrayRef normalized_shape,
                                        const at::Tensor &mean, const at::Tensor &variance,
                                        const c10::optional<at::Tensor> &gamma, const c10::optional<at::Tensor> &beta,
                                        std::array<bool, 3> output_mask)
{
    const int normalized_ndim = static_cast<int>(normalized_shape.size());
    const auto input_shape = X.sizes();
    const auto input_ndim = X.dim();

    TORCH_CHECK(
        (input_ndim >= normalized_ndim && input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)),
        "Given normalized_shape=", normalized_shape, ", expected input with shape [*",
        op_infer::array_to_small_vector(normalized_shape), "], but got input of size", input_shape,
        OPS_ERROR(ErrCode::PARAM));

    const int axis = input_ndim - normalized_ndim;
    const int64_t M =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, 1LL, std::multiplies<int64_t>());
    const int64_t N = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1LL, std::multiplies<int64_t>());
    return layer_norm_backward_npu_support(d_y, X, mean, variance, gamma, M, N, output_mask);
}

tensor_list3 npu_layernorm_grad(const at::Tensor &d_y, const at::Tensor &X, at::IntArrayRef normalized_shape,
                                const at::Tensor &mean, const at::Tensor &variance,
                                const c10::optional<at::Tensor> &gamma, const c10::optional<at::Tensor> &beta)
{
    std::array<bool, 3> output_mask = {true, true, true};
    return acl_op::native_layer_norm_backward(d_y, X, normalized_shape, mean, variance, gamma, beta, output_mask);
}
} // namespace acl_op
