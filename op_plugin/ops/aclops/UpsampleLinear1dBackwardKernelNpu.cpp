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

namespace {
inline void upsample_linear1d_backward_check(const at::Tensor &grad_output, at::IntArrayRef output_size,
                                             at::IntArrayRef input_size)
{
    TORCH_CHECK(output_size.size() == 1, "It is expected output_size equals to 1, but got size ", output_size.size(),
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(input_size.size() == 3, "It is expected input_size equals to 3, but got size ", input_size.size(),
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(grad_output.dim() >= 3, "grad_output dim must larger than 3 ", grad_output.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    int64_t output_width = grad_output.size(2);
    int64_t input_width = input_size[2];

    TORCH_CHECK(output_width > 0 && input_width > 0,
                "Input and output sizes should be greater than 0, but got input (W: ", input_width,
                ") and output (W: ", output_width, ")" + OPS_ERROR(ErrCode::VALUE));
}

at::Tensor &upsample_linear1d_backward_out_nocheck(at::Tensor &result, const at::Tensor &grad_output,
                                                   at::IntArrayRef input_size, bool align_corners,
                                                   c10::optional<double> scales)
{
    c10::SmallVector<float, N> sc = {};
    TORCH_CHECK(input_size.size() == 3 && input_size[2] != 0, "It is expected input_size equals to 3, but got size ",
                input_size.size(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(grad_output.dim() >= 3, "grad_output dim must larger than 3 ", grad_output.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    if (scales.has_value()) {
        sc.push_back(scales.value());
    } else {
        float temp = float(grad_output.size(3)) / float(input_size[2]);
        sc.push_back(temp);
    }
    string coordinate_transformation_mode = align_corners ? "align_corners" : "half_pixel";

    at_npu::native::OpCommand cmd;
    cmd.Name("ResizeGradD")
        .Input(grad_output, "grads")
        .Output(result, "y")
        .Attr("original_size", input_size)
        .Attr("scales", sc)
        .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
        .Attr("mode", static_cast<string>("linear"))
        .Run();
    return result;
}
} // namespace

at::Tensor upsample_linear1d_backward(const at::Tensor &grad_output, at::IntArrayRef output_size,
                                      at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales)
{
    upsample_linear1d_backward_check(grad_output, output_size, input_size);
    at::Tensor grad_output_cp = grad_output;
    if (grad_output.scalar_type() != at::ScalarType::Float) {
        grad_output_cp = at_npu::native::custom_ops::npu_dtype_cast(grad_output_cp, at::ScalarType::Float);
    }
    int64_t N = grad_output_cp.size(0);
    int64_t C = grad_output_cp.size(1);
    int64_t W = input_size[2];
    c10::SmallVector<int64_t, SIZE> output_sizes = {N, C, W};

    // Since only NCHW format input is currently supported, first convert the
    // input grad_output (3 dimensions) to 4 dimensions as the input of npu
    auto grad_output_4dim = grad_output_cp.unsqueeze(2);

    at::Tensor result = npu_preparation::apply_tensor(grad_output_cp, output_sizes);
    upsample_linear1d_backward_out_nocheck(result, grad_output_4dim, input_size, align_corners, scales);

    if (result.dtype() != grad_output.dtype()) {
        result = result.to(grad_output.dtype());
    }

    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_linear1d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input_size.size() == 3,
        "It is expected input_size equals to 3, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);

    return acl_op::upsample_linear1d_backward(grad_output, osize, input_size, align_corners, scales_w);
}
#endif
} // namespace acl_op
