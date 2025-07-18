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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list3 = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

static inline c10::SmallVector<int64_t, op_infer::N> expand_dim_if_needed(
    at::IntArrayRef list_param,
    const char *param_name,
    int64_t expected_dim)
{
    if (list_param.size() == 1) {
        c10::SmallVector<int64_t, op_infer::N> expand_dim_param_vec;
        for (int64_t i = 0; i < expected_dim; i++) {
            expand_dim_param_vec.emplace_back(list_param[0]);
        }
        return expand_dim_param_vec;
    } else {
        return op_plugin::utils::convert_array_to_vector(list_param);
    }
}

static tensor_list3 _calc_convolution_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                               const at::Tensor &weight, c10::optional<at::IntArrayRef> bias_sizes_opt,
                                               at::IntArrayRef stride, at::IntArrayRef padding,
                                               at::IntArrayRef dilation, bool transposed,
                                               at::IntArrayRef output_padding, int64_t groups,
                                               ::std::array<bool, 3> output_mask)
{
    int64_t k = weight.ndimension();
    int64_t dim = k - 2;
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_calc_convolution_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if ((is_allow_internel_format || is_jit_enable) && (dim != 3)) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expand =
        expand_dim_if_needed(output_padding, "output_padding", dim);
    output_padding = at::IntArrayRef(output_padding_expand);

    auto outputSizes =
        op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    if (output_mask[2]) {
        gradBias = npu_preparation::apply_tensor_without_format(*bias_sizes_opt, grad_output.options());
    } else {
        gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());
    }

    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                 transposed, output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);

    FLOP_COUNT(FlopCounter::conv_backward_flop, grad_output, input, weight, transposed, output_mask, gradInput, gradWeight);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

// length of output_mask is 3
tensor_list3 convolution_backward(const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &weight,
                                  c10::optional<at::IntArrayRef> bias_sizes_opt, at::IntArrayRef stride,
                                  at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                                  at::IntArrayRef output_padding, int64_t groups, ::std::array<bool, 3> output_mask)
{
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    op_plugin::utils::check_input_same_type_as_parameters(input, weight);
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_tbc_backward(const at::Tensor &self, const at::Tensor &input,
                                                                 const at::Tensor &weight, const at::Tensor &bias,
                                                                 int64_t pad)
{
    DO_COMPATIBILITY(aclnnConvTbcBackward, acl_op::conv_tbc_backward(self, input, weight, bias, pad));
    // construct other inputs of the NPU
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    at::IntArrayRef stride = {1, 1};
    at::IntArrayRef padding = {0, pad};
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    std::array<bool, 3> grad_input_mask = {1, 1, 1};
    // calculate outputSizes of every output
    auto outputSizes =
        op_infer::conv2d_backward_tbc_output_size(input, self, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    at::Tensor gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    at::Tensor gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), self.options());
    // execute hostapi
    EXEC_NPU_CMD(aclnnConvTbcBackward, self, input, weight, bias, pad, cube_math_type, gradInput, gradWeight, gradBias);
    return std::make_tuple(gradInput, gradWeight, gradBias);
}

tensor_list3 slow_conv_transpose2d_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                            const at::Tensor &weight, at::IntArrayRef kernel_size,
                                            at::IntArrayRef stride, at::IntArrayRef padding,
                                            at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                            std::array<bool, 3> output_mask)
{
    int64_t groups = 1;
    bool transposed = true;
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

tensor_list3 slow_conv_dilated2d_backward(const at::Tensor &grad_output, const at::Tensor &input,
                                          const at::Tensor &weight, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                          at::IntArrayRef padding, at::IntArrayRef dilation,
                                          std::array<bool, 3> output_mask)
{
    at::IntArrayRef output_padding = {0, 0};
    int64_t groups = 1;
    bool transposed = true;
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(const at::Tensor &grad_output,
                                                                     const at::Tensor &input, const at::Tensor &weight,
                                                                     at::IntArrayRef kernel_size,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding,
                                                                     std::array<bool, 3> output_mask)
{
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    bool transposed = false;
    at::IntArrayRef output_padding = {0, 0};
    auto w_sizes = weight.sizes();
    at::Tensor orig_weight = npu_preparation::apply_tensor_without_format(w_sizes, weight.options());
    int64_t s1 = w_sizes[0];
    // get the product of w_sizes from the 1 index to the last
    int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
    TORCH_CHECK(kernel_size[0] * kernel_size[1] != 0, "kernel_size should not be zero", OPS_ERROR(ErrCode::PARAM));
    s2 = s2 / (kernel_size[0] * kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> slow_weight_size = {s1, s2, kernel_size[0], kernel_size[1]};
    weight.resize_(slow_weight_size);
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));

    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    /* k == 5 and groups > 3 currently unsupported by the binary file
      CheckForbidInternalFormat = False: turn on private format��CheckJitDisable = False: turn on JitCompile
    */
    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_slow_conv2d_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if (is_allow_internel_format || is_jit_enable) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    auto outputSizes =
        op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());
    gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());

    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                 transposed, output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);
    auto orig_sizes = orig_weight.sizes();
    gradWeight.resize_(orig_sizes);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

tensor_list3 convolution_backward_overrideable(const at::Tensor &grad_output, const at::Tensor &input,
                                               const at::Tensor &weight, c10::IntArrayRef stride,
                                               c10::IntArrayRef padding, c10::IntArrayRef dilation, bool transposed,
                                               c10::IntArrayRef output_padding, int64_t groups,
                                               std::array<bool, 3> output_mask)
{
    c10::optional<at::IntArrayRef> bias_sizes_opt = {grad_output.size(1)};
    DO_COMPATIBILITY(aclnnConvolutionBackward,
                     acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                                  transposed, output_padding, groups, output_mask));
    return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation, transposed,
                                      output_padding, groups, output_mask);
}

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
static std::tuple<at::Tensor, at::Tensor, at::Tensor> _calc_convolution_backward(
    const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight,
    const at::OptionalIntArrayRef bias_sizes_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    ::std::array<bool, 3> output_mask)
{
    int64_t k = weight.ndimension();
    int64_t dim = k - 2;
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_calc_convolution_backward exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    // CheckForbidInternalFormat = False: turn on private format��CheckJitDisable = False: turn on JitCompile
    if ((is_allow_internel_format || is_jit_enable)) {
        return acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                            transposed, output_padding, groups, output_mask);
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim_if_needed(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim_if_needed(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim_if_needed(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expand = expand_dim_if_needed(output_padding, "output_padding",
                                                                                        dim);
    output_padding = at::IntArrayRef(output_padding_expand);

    auto outputSizes = op_infer::conv2d_backward_npu_output_size(input, grad_output, weight);

    // construct the output tensor of the NPU
    at::Tensor gradInput;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    gradInput = npu_preparation::apply_tensor_without_format(std::get<0>(outputSizes), input.options());
    gradWeight = npu_preparation::apply_tensor_without_format(std::get<1>(outputSizes), weight.options());

    // use 2nd dimension of outputSizes
    gradBias = npu_preparation::apply_tensor_without_format(std::get<2>(outputSizes), grad_output.options());

    int64_t input_dim = input.ndimension();
    at::optional<c10::IntArrayRef> bias_sizes = c10::nullopt;
    if (bias_sizes_opt.has_value()) {
        bias_sizes = bias_sizes_opt.value();
    }
    EXEC_NPU_CMD(aclnnConvolutionBackward, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
                 output_padding, groups, output_mask, cube_math_type, gradInput, gradWeight, gradBias);
    return std::make_tuple(std::move(gradInput), std::move(gradWeight), std::move(gradBias));
}

// length of output_mask is 3
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  DO_COMPATIBILITY(aclnnConvolutionBackward, acl_op::convolution_backward(grad_output, input, weight, bias_sizes_opt,
                                                                          stride, padding, dilation, transposed,
                                                                          output_padding, groups, output_mask));
  return _calc_convolution_backward(grad_output, input, weight, bias_sizes_opt, stride, padding, dilation,
                                    transposed, output_padding, groups, output_mask);
}
#endif
}
