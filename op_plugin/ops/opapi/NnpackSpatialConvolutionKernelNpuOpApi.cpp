// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int AXIS_THREE = 3;
const int AXIS_TWO = 2;

at::Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride)
{
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    auto output_size = op_infer::nnpack_spatial_convolution_npu_output_size(input, weight, padding, stride);
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[0], padding[0]};
    c10::SmallVector<int64_t, N> strides_size = {stride[0], stride[0]};
    if (padding.size() != 1) {
        paddings[AXIS_TWO] = padding[1];
        paddings[AXIS_THREE] = padding[1];
    }
    if (stride.size() != 1) {
        strides_size[1] = stride[1];
    }
    c10::SmallVector<int64_t, N> dilations = {1, 1};
    c10::SmallVector<int64_t, N> output_padding = {0, 0};
    at::IntArrayRef paddings_intarray = at::IntArrayRef(paddings);
    at::IntArrayRef strides_size_intarray = at::IntArrayRef(strides_size);
    at::IntArrayRef dilations_intarray = at::IntArrayRef(dilations);
    at::IntArrayRef output_padding_intarray = at::IntArrayRef(output_padding);
    int64_t groups = 1;
    bool transposed = false;
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, input.options());
    EXEC_NPU_CMD(aclnnConvolution, input, weight, bias, strides_size_intarray, paddings_intarray, dilations_intarray,
        transposed, output_padding_intarray, groups, result, cube_math_type);
    return result;
}
} // namespace op_api
