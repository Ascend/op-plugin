// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

static inline c10::SmallVector<int64_t, op_infer::N> expand_dim(at::IntArrayRef list_param, const char *param_name,
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

at::Tensor convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                       at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                       at::IntArrayRef output_padding, int64_t groups)
{
    return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false,
                            false, false, false);
}

static at::Tensor _calc_convolution(const at::Tensor &input, const at::Tensor &weight,
                                    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                                    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                                    at::IntArrayRef output_padding, int64_t groups)
{
    int64_t k = weight.ndimension();
    int64_t inputK = input.ndimension();
    int64_t dim = k - 2; // Subtract nonspatial dimensions: 2
    bool unBatch = false;

    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    // CheckForbidInternalFormat = False: turn on private format；CheckJitDisable = False: turn on JitCompile
    ASCEND_LOGI("_calc_convolution exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if ((is_allow_internel_format || is_jit_enable) && (dim != 3)) {
        return acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                                    false, false, false, false);
    }
    // Conv1D: 1, Conv2D: 2, Conv3D: 3
    if (dim != 1 && dim != 2 && dim != 3) {
        return at::Tensor();
    }

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expend = expand_dim(output_padding, "output_padding", dim);
    output_padding = at::IntArrayRef(output_padding_expend);

    c10::SmallVector<int64_t, SIZE> out_size;

    out_size = op_infer::conv_npu_output_size(input, weight, bias, padding, output_padding, stride, dilation, groups,
                                              transposed);

    auto output = npu_preparation::apply_tensor_without_format(out_size, input.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    EXEC_NPU_CMD(aclnnConvolution, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                 output, cube_math_type);

    // input dim = 3 while conv2D: 2
    if (dim == 2 && inputK == 3) {
        c10::SmallVector<int64_t, SIZE> squeeze_size = {output.size(1), output.size(2), output.size(3)};
        output.resize_(squeeze_size);

        c10::SmallVector<int64_t, SIZE> squeeze_size_input = {input.size(1), input.size(2), input.size(3)};
        input.resize_(squeeze_size_input);
    }

    FLOP_COUNT(FlopCounter::conv_flop, input, weight, transposed, output);
    return output;
}

at::Tensor _convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                        at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                        at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic,
                        bool cudnn_enabled, bool allow_tf32)
{
    DO_COMPATIBILITY(aclnnConvolution,
                     acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                          groups, benchmark, deterministic, cudnn_enabled, allow_tf32));
    const at::Tensor &bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });
    at::Tensor bias_val = bias_opt;
    op_plugin::utils::check_input_same_type_as_parameters(input, weight, bias_val);
    return _calc_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

at::Tensor slow_conv_transpose2d(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                 const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                 at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    bool transposed = true;
    int64_t groups = 1;
    DO_COMPATIBILITY(aclnnConvolution, acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                                                            output_padding, groups, false, false, false, false));

    return _calc_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

at::Tensor &slow_conv_transpose2d_out(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                      const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                      at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                      at::Tensor &output)
{
    DO_COMPATIBILITY(aclnnConvolution, acl_op::slow_conv_transpose2d_out(input, weight, kernel_size, bias_opt, stride,
                                                                         padding, output_padding, dilation, output));

    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    bool transposed = true;
    int64_t groups = 1;
    int64_t k = weight.ndimension();
    int64_t dim = k - 2; // Subtract nonspatial dimensions: 2

    c10::SmallVector<int64_t, op_infer::N> stride_expand = expand_dim(stride, "stride", dim);
    stride = at::IntArrayRef(stride_expand);

    c10::SmallVector<int64_t, op_infer::N> padding_expand = expand_dim(padding, "padding", dim);
    padding = at::IntArrayRef(padding_expand);

    c10::SmallVector<int64_t, op_infer::N> dilation_expand = expand_dim(dilation, "dilation", dim);
    dilation = at::IntArrayRef(dilation_expand);

    c10::SmallVector<int64_t, op_infer::N> output_padding_expend = expand_dim(output_padding, "output_padding", dim);
    output_padding = at::IntArrayRef(output_padding_expend);

    c10::SmallVector<int64_t, SIZE> out_size;

    out_size = op_infer::conv_npu_output_size(input, weight, bias, padding, output_padding, stride, dilation, groups,
                                              transposed);

    if (bias.defined()) {
        npu_preparation::check_tensor({input, weight, bias}, {output}, input.scalar_type(), out_size);
    } else {
        npu_preparation::check_tensor({input, weight}, {output}, input.scalar_type(), out_size);
    }

    int64_t inputK = input.ndimension();
    bool unBatch = false;

    // Groups > 1 and 3D scenes are currently not supported (binary operator problem), and path 3 implementation is
    // temporarily called
    // CheckForbidInternalFormat = False: turn on private format; CheckJitDisable = False: turn on JitCompile
    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("slow_conv_transpose2d_out exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if ((is_allow_internel_format || is_jit_enable)) {
        output = acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                      groups, false, false, false, false);
        return output;
    }
    // Conv1D: 1, Conv2D: 2
    if (dim != 1 && dim != 2 && dim != 3) {
        output = at::Tensor();
        return output;
    }
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    EXEC_NPU_CMD(aclnnConvolution, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                 output, cube_math_type);

    // input dim = 3 while conv2D: 2
    if (dim == 2 && inputK == 3) {
        c10::SmallVector<int64_t, SIZE> squeeze_size = {output.size(1), output.size(2), output.size(3)};
        output.resize_(squeeze_size);

        c10::SmallVector<int64_t, SIZE> squeeze_size_input = {input.size(1), input.size(2), input.size(3)};
        input.resize_(squeeze_size_input);
    }
    return output;
}

at::Tensor slow_conv_dilated2d(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                               const c10::optional<at::Tensor> &bias, at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation)
{
    bool transposed = false;
    at::IntArrayRef output_padding = {0, 0};
    int64_t groups = 1;
    DO_COMPATIBILITY(aclnnConvolution, acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                                                            output_padding, groups, false, false, false, false));

    return _calc_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

at::Tensor _slow_conv2d_forward(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                at::IntArrayRef padding)
{
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    bool transposed = false;
    at::IntArrayRef output_padding = {0, 0};
    auto w_sizes = weight.sizes();
    int64_t s1 = w_sizes[0];
    // get the product of w_sizes from the 1 index to the last
    int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
    TORCH_CHECK(kernel_size[0] * kernel_size[1] != 0, "kernel_size should not be zero", OPS_ERROR(ErrCode::PARAM));
    s2 = s2 / (kernel_size[0] * kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> slow_weight_size = {s1, s2, kernel_size[0], kernel_size[1]};
    weight.resize_(slow_weight_size);
    DO_COMPATIBILITY(aclnnConvolution, acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                                                            output_padding, groups, false, false, false, false));
    return _calc_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

at::Tensor &_slow_conv2d_forward_out(const at::Tensor &input, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                     const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                     at::IntArrayRef padding, at::Tensor &output)
{
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;
    at::IntArrayRef dilation = {1, 1};
    int64_t groups = 1;
    bool transposed = false;
    at::IntArrayRef output_padding = {0, 0};
    auto w_sizes = weight.sizes();
    int64_t s1 = w_sizes[0];
    // get the product of w_sizes from the 1 index to the last
    int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
    TORCH_CHECK(kernel_size[0] * kernel_size[1] != 0, "kernel_size should not be zero", OPS_ERROR(ErrCode::PARAM));
    s2 = s2 / (kernel_size[0] * kernel_size[1]);
    c10::SmallVector<int64_t, SIZE> slow_weight_size = {s1, s2, kernel_size[0], kernel_size[1]};
    weight.resize_(slow_weight_size);

    DO_COMPATIBILITY(aclnnConvolution,
                     acl_op::_slow_conv2d_forward_out(input, weight, kernel_size, bias, stride, padding, output));

    c10::SmallVector<int64_t, SIZE> out_size;
    out_size = op_infer::conv_npu_output_size(input, weight, bias, padding, output_padding, stride, dilation, groups,
                                              transposed);

    if (bias.defined()) {
        npu_preparation::check_tensor({input, weight, bias}, {output}, input.scalar_type(), out_size);
    } else {
        npu_preparation::check_tensor({input, weight}, {output}, input.scalar_type(), out_size);
    }

    // temporarily called
    // CheckForbidInternalFormat = False: turn on private format; CheckJitDisable = False: turn on JitCompile
    bool is_jit_enable = !at_npu::native::env::CheckJitDisable();
    bool is_allow_internel_format = !at_npu::native::env::CheckForbidInternalFormat();
    ASCEND_LOGI("_slow_conv2d_forward_out exec with jit compile: %d, allow internal format: %d",
                is_jit_enable, is_allow_internel_format);
    if (is_allow_internel_format || is_jit_enable) {
        output = acl_op::_slow_conv2d_forward_out(input, weight, kernel_size, bias, stride, padding, output);
        return output;
    }

    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    EXEC_NPU_CMD(aclnnConvolution, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
                 output, cube_math_type);

    return output;
}

at::Tensor convolution_overrideable(const at::Tensor &input, const at::Tensor &weight,
                                    const c10::optional<at::Tensor> &bias, c10::IntArrayRef stride,
                                    c10::IntArrayRef padding, c10::IntArrayRef dilation, bool transposed,
                                    c10::IntArrayRef output_padding, int64_t groups)
{
    DO_COMPATIBILITY(aclnnConvolution, acl_op::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                                                            output_padding, groups, false, false, false, false));
    return _calc_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}


} // namespace op_api
