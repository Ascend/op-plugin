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

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(const at::Tensor &input,
                                                                 at::IntArrayRef normalized_shape,
                                                                 const c10::optional<at::Tensor> &weight_ex,
                                                                 const c10::optional<at::Tensor> &bias_ex, double eps)
{
    DO_COMPATIBILITY(aclnnLayerNorm, acl_op::native_layer_norm(input, normalized_shape, weight_ex, bias_ex, eps));
    const at::Tensor &weight_op = c10::value_or_else(weight_ex, [] { return at::Tensor(); });
    const at::Tensor &bias_op = c10::value_or_else(bias_ex, [] { return at::Tensor(); });
    at::Tensor weight =
        weight_op.defined() ? weight_op.resize_(normalized_shape) : at::ones(normalized_shape, input.options());
    at::Tensor bias =
        bias_op.defined() ? bias_op.resize_(normalized_shape) : at::zeros(normalized_shape, input.options());

    // construct output for hostapi
    auto output = at_npu::native::OpPreparation::apply_tensor_without_format(input);
    at::Tensor mean_out;
    at::Tensor rstd_out;

    const size_t norm_ndim = normalized_shape.size();
    const auto input_ndim = input.dim();
    const size_t begin_axis = input_ndim - norm_ndim;

    const auto input_shape = input.sizes();
    const int64_t M =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());
    auto acc_type = input.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;

    // shape and dtype of mean and rstd depend on M value and input dtype
    if (M <= 0) {
        mean_out = at_npu::native::OpPreparation::apply_tensor_without_format({M}, input.options().dtype(acc_type));
        rstd_out = at_npu::native::OpPreparation::apply_tensor_without_format({M}, input.options().dtype(acc_type));
    } else {
        at::SmallVector<int64_t, 8> mean_shape;
        for (size_t index = 0; index < begin_axis; index++) {
            mean_shape.emplace_back(input.size(index));
        }
        for (size_t index = begin_axis; index < input_ndim; index++) {
            mean_shape.emplace_back(1);
        }
        mean_out =
            at_npu::native::OpPreparation::apply_tensor_without_format(mean_shape, input.options().dtype(acc_type));
        rstd_out =
            at_npu::native::OpPreparation::apply_tensor_without_format(mean_shape, input.options().dtype(acc_type));
    }
    // call HostAPI function
    EXEC_NPU_CMD(aclnnLayerNorm, input, normalized_shape, weight, bias, eps, output, mean_out, rstd_out);
    return std::tie(output, mean_out, rstd_out);
}

} // namespace op_api
