// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
bool is_special_conv1d(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups)
{
    if (stride[1] > 63 && stride[1] == weight.size(3) && padding[1] == 0 && dilation[1] == 1 && groups == 1 &&
        input.size(1) == 1) {
        return true;
    } else {
        return false;
    }
}
} // namespace

at::Tensor &npu_conv2d_out(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                           at::Tensor &result)
{
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2D").Input(input, "x").Input(weight, "filter");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(result, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", (string) "NCHW")
        .Run();

    return result;
}

at::Tensor npu_conv2d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                      at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups)
{
    TORCH_CHECK(input.dim() >= 4, "input has to be more than 4D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 4, "weight has to more than 4D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.size(3) != 0, "4th dim of weight cannot be 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));

    // support special scenario
    if (is_special_conv1d(input, weight, stride, padding, dilation, groups)) {
        at::Tensor mm_input = input.view({input.size(0), input.size(3) / weight.size(3), weight.size(3)});
        at::Tensor mm_other = weight.view({weight.size(0), weight.size(3)}).permute({1, 0});
        at::Tensor mm_result = at::matmul(mm_input, mm_other);
        at::Tensor result = mm_result.permute({0, 2, 1}).unsqueeze(2);
        return result;
    }

    int64_t N = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t Co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);

    int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;

    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho,
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo,
        OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Ho, Wo};
    int64_t result_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
    at::Tensor result = npu_preparation::apply_tensor_with_format(input, output_size, result_format);
    acl_op::npu_conv2d_out(input, weight, bias, stride, padding, dilation, groups, result);
    return result;
}
} // namespace acl_op
