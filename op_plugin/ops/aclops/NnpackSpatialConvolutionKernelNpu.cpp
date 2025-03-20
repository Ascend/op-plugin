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
at::Tensor _nnpack_spatial_convolution_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef padding,
    at::IntArrayRef stride)
{
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[0], padding[0]};
    c10::SmallVector<int64_t, N> strides_size = {1, 1, stride[0], stride[0]};
    if (padding.size() != 1) {
        paddings[2] = padding[1];
        paddings[3] = padding[1];
    }
    if (stride.size() != 1) {
        strides_size[3] = stride[1];
    }
    c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
    string data_format = "NCHW";
    int64_t groups = 1;

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2D")
        .Input(input)
        .Input(weight)
        .Input(bias)
        .Output(result)
        .Attr("strides", strides_size)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("groups", groups)
        .Attr("data_format", data_format)
        .Run();
    return result;
}
} // namespace

at::Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride)
{
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    auto output_size = op_infer::nnpack_spatial_convolution_npu_output_size(input, weight, padding, stride);
    int64_t result_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, input.options(), result_format);
    _nnpack_spatial_convolution_npu_nocheck(result, input, weight, bias, padding, stride);
    return result;
}
} // namespace acl_op
