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

#if VERSION_BETWEEN(V2R1, V2R6)
const at::Tensor &_conv_depthwise2d_out(const at::Tensor &self, const at::Tensor &weight, c10::IntArrayRef kernel_size,
                                        const c10::optional<at::Tensor> &bias_opt, c10::IntArrayRef stride,
                                        c10::IntArrayRef padding, c10::IntArrayRef dilation, const at::Tensor &result)
{
    TORCH_CHECK(weight.dim() >= 4, "weight has to be more than 4D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor &weight_modify = weight.permute({1, 0, 2, 3});

    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    at::Tensor temp_out = result;

    at_npu::native::OpCommand cmd;
    cmd.Name("DepthwiseConv2D").Input(self, "x").Input(weight_modify, "filter");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(temp_out, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("data_format", (string) "NCHW")
        .Run();
    return result;
}
#endif

#if VERSION_BETWEEN(V2R7, VERSION_NEWEST)
at::Tensor &_conv_depthwise2d_out(const at::Tensor &self, const at::Tensor &weight, c10::IntArrayRef kernel_size,
                                  const c10::optional<at::Tensor> &bias_opt, c10::IntArrayRef stride,
                                  c10::IntArrayRef padding, c10::IntArrayRef dilation, at::Tensor &result)
{
    TORCH_CHECK(weight.dim() >= 4, "weight has to be more than 4D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor &weight_modify = weight.permute({1, 0, 2, 3});

    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[1]};
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
    c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
    at::Tensor temp_out = result;

    at_npu::native::OpCommand cmd;
    cmd.Name("DepthwiseConv2D").Input(self, "x").Input(weight_modify, "filter");
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(temp_out, "y")
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("data_format", (string) "NCHW")
        .Run();
    return result;
}
#endif

at::Tensor _conv_depthwise2d(const at::Tensor &self, const at::Tensor &weight, c10::IntArrayRef kernel_size,
                             const c10::optional<at::Tensor> &bias_opt, c10::IntArrayRef stride,
                             c10::IntArrayRef padding, c10::IntArrayRef dilation)
{
    TORCH_CHECK(self.dim() >= 4, "self has to be more than 4D, but got Tensor of dimension ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size.size() >= 2, "kernel_size has to contain more than 2 elements, but got ",
        kernel_size.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 2, "stride has to contain more than 2 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= 2, "padding has to contain more than 2 elements, but got ", padding.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 2, "dilation has to contain more than 2 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride[0] * stride[1] != 0, "Stride cannot contain 0" + OPS_ERROR(ErrCode::PARAM));
    int64_t N = self.size(0);
    int64_t Co = weight.size(0);
    int64_t H = self.size(2);
    int64_t W = self.size(3);
    int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;

    TORCH_CHECK(Ho > 0, "Ho has to be positive, but got ", Ho,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(Wo > 0, "Wo has to be positive, but got ", Wo,
        OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {N, Co, Ho, Wo};
    int64_t result_format = self.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
    at::Tensor result = npu_preparation::apply_tensor_with_format(self, output_size, result_format);
    return acl_op::_conv_depthwise2d_out(self, weight, kernel_size, bias_opt, stride, padding, dilation, result);
}
} // namespace acl_op
