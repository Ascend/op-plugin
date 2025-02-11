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

#include <c10/core/GradMode.h>
#include <ATen/native/ConvUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {

namespace {
constexpr int input_batch_size_dim = 0;
constexpr int output_batch_size_dim = 0;
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ConvParams {
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    bool transposed;
    std::vector<int64_t> output_padding;
    int groups;
    bool benchmark;
    bool deterministic;
    bool allow_tf32;

    bool is_dilated() const;
    bool is_output_padding_neg() const;
    bool is_padding_neg() const;
    bool is_stride_nonpos() const;
    void view1d_as_2d();
};

auto ConvParams::is_dilated() const -> bool
{
    bool is_dilated = false;
    for (auto d : dilation) {
        is_dilated |= (d != 1);
    }
    return is_dilated;
}

auto ConvParams::is_output_padding_neg() const -> bool
{
    bool is_non_neg = false;
    for (auto p : output_padding) {
        is_non_neg |= (p < 0);
    }
    return is_non_neg;
}

auto ConvParams::is_padding_neg() const -> bool
{
    bool is_non_neg = false;
    for (auto p : padding) {
        is_non_neg |= (p < 0);
    }
    return is_non_neg;
}

auto ConvParams::is_stride_nonpos() const -> bool
{
    bool is_nonpos = false;
    for (auto s : stride) {
        is_nonpos |= (s <= 0);
    }
    return is_nonpos;
}

auto ConvParams::view1d_as_2d() -> void
{
    if (stride.size() == 1) {
        stride.insert(stride.begin(), 1);
        padding.insert(padding.begin(), 0);
        dilation.insert(dilation.begin(), 1);
        output_padding.insert(output_padding.begin(), 0);
    }
}

void view1d_as_2d(c10::SmallVector<int64_t, N> &stride, c10::SmallVector<int64_t, N> &padding,
                  c10::SmallVector<int64_t, N> &dilation, c10::SmallVector<int64_t, N> &output_padding)
{
    if (stride.size() == 1) {
        stride.insert(stride.begin(), 1);
        padding.insert(padding.begin(), 0);
        dilation.insert(dilation.begin(), 1);
        output_padding.insert(output_padding.begin(), 0);
    }
}

at::Tensor view4d(const at::Tensor &tensor)
{
    return tensor.unsqueeze(2);
}

at::Tensor view3d(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.ndimension() == 4, "expected 4D tensor, got tensor with ", tensor.ndimension(),
                " dimensions instead" + OPS_ERROR(ErrCode::PARAM));
    return tensor.squeeze(2);
}

std::tuple<at::Tensor, bool> batchify(
    const at::Tensor& input,
    const int64_t num_spatial_dims,
    const std::string& func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = (input.dim() == dim_count_batch);
  TORCH_CHECK(input.dim() == dim_count_no_batch || is_batched,
      "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch,
      "D (batched) input to ", func_name, ", but got input of size: ", input.sizes(), OPS_ERROR(ErrCode::PARAM));
  return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
}

inline std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    TORCH_CHECK(false, "expected ", param_name, " to be a single integer value or a list of ", expected_dim,
                " values to match the convolution dimensions, but got ", param_name, "=", list_param, OPS_ERROR(ErrCode::PARAM));
  } else {
    return list_param.vec();
  }
}

inline c10::SmallVector<int64_t, N> expand_dim_if_needed(at::IntArrayRef list_param, const char *param_name,
                                                         int64_t expected_dim)
{
    if (list_param.size() == 1) {
        c10::SmallVector<int64_t, N> expand_dim_param_vec;
        for (int64_t i = 0; i < expected_dim; i++) {
            expand_dim_param_vec.emplace_back(list_param[0]);
        }
        return expand_dim_param_vec;
    } else {
        return op_plugin::utils::convert_array_to_vector(list_param);
    }
}
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
void check_shape_forward(const at::Tensor &input, const c10::IntArrayRef &weight_sizes, const at::Tensor &bias,
                         const ConvParams &params)
{
    int64_t k = input.ndimension();
    int64_t weight_dim = static_cast<int64_t>(weight_sizes.size());
    int64_t groups = params.groups;
    const auto &padding = params.padding;
    const auto &dilation = params.dilation;
    bool transposed = params.transposed;

    TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported" + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported" + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported" + OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(weight_dim == k, "Expected ", weight_dim, "-dimensional input for ", weight_dim, "-dimensional weight ",
                weight_sizes, ", but got ", k, "-dimensional input of size ", input.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(padding.size() >= k - 2, "padding must be smaller than the number of dimensions", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= k - 2, "dilation must be smaller than the number of dimensions", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_sizes[0] >= groups, "Given groups=", groups, ", expected weight to be at least ", groups,
                " at dimension 0, but got weight of size ", weight_sizes, " instead", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(groups != 0, "groups should not be zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_sizes[0] % groups == 0, "Given groups=", groups, ", expected weight to be divisible by ", groups,
                " at dimension 0, but got weight of size [", weight_sizes, "] instead", OPS_ERROR(ErrCode::PARAM));

    if (!transposed) {
        std::vector<int64_t> input_shape;
        std::vector<int64_t> kernel_shape;
        bool kernel_size_correct = true;

        TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups), "Given groups=", groups, ", weight of size ",
                    weight_sizes, ", expected input", input.sizes(), " to have ", (weight_sizes[1] * groups),
                    " channels, but got ", input.size(1), " channels instead", OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
                    "Given weight of size ", weight_sizes, ", expected bias to be 1-dimensional with ", weight_sizes[0],
                    " elements", ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));

        for (const auto i : c10::irange(2, k)) {
            input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
            // log new kernel size considering dilation
            kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
            if (input_shape.back() < kernel_shape.back()) {
                kernel_size_correct = false;
            }
        }

        TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel", OPS_ERROR(ErrCode::PARAM));

        if (!kernel_size_correct) {
            // If kernel size is incorrect
            std::ostringstream input_ss;
            std::ostringstream kernel_ss;
            std::string separator = "";

            for (size_t i = 0, len = input_shape.size(); i < len; ++i) {
                input_ss << separator << input_shape[i];
                kernel_ss << separator << kernel_shape[i];
                separator = " x ";
            }

            TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). Kernel size: (",
                        kernel_ss.str(), "). Kernel size can't be greater than actual input size", OPS_ERROR(ErrCode::PARAM));
        }
    } else {
        // transposed
        TORCH_CHECK(input.size(1) == weight_sizes[0], "Given transposed=", transposed, ", weight of size ",
                    weight_sizes, ", expected input", input.sizes(), " to have ", weight_sizes[0],
                    " channels, but got ", input.size(1), " channels instead", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
                    "Given transposed=", transposed, ", weight of size ", weight_sizes,
                    ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
                    ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
    }
}

void check_shape_backward(const at::Tensor &input, const c10::IntArrayRef &weight_sizes, const ConvParams &params)
{
    check_shape_forward(input, weight_sizes, at::Tensor(), params);
}

at::native::ConvBackend select_conv_backend(const at::Tensor &input, const at::Tensor &weight,
                                            const c10::optional<at::IntArrayRef> bias_sizes_opt,
                                            const bool need_backward, const ConvParams &params)
{
    // don't send empty inputs through backends
    if (input.size(0) == 0 || input.size(1) == 0) {
        return at::native::ConvBackend::Empty;
    } else if (input.numel() == 0) {
        TORCH_CHECK(false,
                    "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes(), OPS_ERROR(ErrCode::PARAM));
    }

    if (torch_npu::utils::is_npu(input)) {
        // backends without support for groups
        if (params.transposed) {
            if (input.ndimension() == 4) {
                return at::native::ConvBackend::SlowTranspose2d;
            } else if (input.ndimension() == 5) {
                return at::native::ConvBackend::SlowTranspose3d;
            } else {
                TORCH_CHECK(false, "Only 4D or 5D input is supported" + OPS_ERROR(ErrCode::PARAM));
            }
        } else { /* Not transposed */
            if (input.ndimension() == 4) {
                if (params.is_dilated()) {
                    return at::native::ConvBackend::SlowDilated2d;
                } else {
                    return at::native::ConvBackend::Slow2d;
                }
            } else if (input.ndimension() == 5) {
                return at::native::ConvBackend::Slow3d;
            } else {
                TORCH_CHECK(false, "Only 4D or 5D input is supported" + OPS_ERROR(ErrCode::PARAM));
            }
        }
    } else {
        // Only reach here when input is backend with out-of-source implementation.
        return at::native::ConvBackend::Overrideable;
    }
    // Error out if no suitable backend was found.
    TORCH_CHECK(false, "unsupported ConvNd parameters" + OPS_ERROR(ErrCode::PARAM));
}

// Selects a backend for convolution based on the inputs and params.
at::native::ConvBackend select_conv_backend(const at::Tensor &input_r, const at::Tensor &weight_r,
                                            const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride_opt,
                                            at::IntArrayRef padding_opt, at::IntArrayRef dilation_opt, bool transposed,
                                            at::IntArrayRef output_padding_opt, int64_t groups)
{
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;

    auto &ctx = at::globalContext();
    auto k = weight_r.ndimension();
    int64_t dim = k - 2;
    ConvParams params;
    params.stride = expand_param_if_needed(stride_opt, "stride", dim);
    params.padding = expand_param_if_needed(padding_opt, "padding", dim);
    params.dilation = expand_param_if_needed(dilation_opt, "dilation", dim);
    params.transposed = transposed;
    params.output_padding = expand_param_if_needed(output_padding_opt, "output_padding", dim);
    params.groups = groups;

    auto input = input_r;
    auto weight = weight_r;
    check_shape_forward(input, weight.sizes(), bias, params);

    // Expand 1d -> 2d.
    // This is only done for backends that don't natively support 1d spatial input.
    if (k == 3 && !input.is_mkldnn()) {
        // avoid accidentally going through NHWC for permuted 3d input.
        params.view1d_as_2d();
        input = view4d(input);
        weight = view4d(weight);
    }

    auto bias_sizes_opt = bias.defined() ? c10::optional<at::IntArrayRef>(bias.sizes()) : c10::nullopt;
    bool need_backward = c10::GradMode::is_enabled() &&
                         (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
    return select_conv_backend(input, weight, bias_sizes_opt, need_backward, params);
}

at::Tensor conv_transpose2d(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                            at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding,
                            int64_t groups, at::IntArrayRef dilation)
{
    return at::convolution(input, weight, bias, stride, padding, dilation, true, output_padding, groups);
}

at::Tensor conv_transpose3d(const at::Tensor &input_opt, const at::Tensor &weight,
                            const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation)
{
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;

    at::Tensor input;
    bool is_batched;
    std::tie(input, is_batched) = batchify(input_opt, 3, "conv_transpose3d");
    auto output = at::convolution(input, weight, bias, stride, padding, dilation, true, output_padding, groups);
    return is_batched ? output : output.squeeze(0);
}

at::Tensor convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias,
                       at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                       at::IntArrayRef output_padding, int64_t groups)
{
    return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false,
                            false, false);
}

at::Tensor _convolution(const at::Tensor &input_opt, const at::Tensor &weight_opt,
                        const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride_opt,
                        at::IntArrayRef padding_opt, at::IntArrayRef dilation_opt, bool transposed,
                        at::IntArrayRef output_padding_opt, int64_t groups, bool benchmark, bool deterministic,
                        bool cudnn_enabled, bool allow_tf32)
{
    at::Tensor input = input_opt;
    at::Tensor weight = weight_opt;

    const at::Tensor &bias_val = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    at::Tensor bias = bias_val;
    op_plugin::utils::check_input_same_type_as_parameters(input, weight, bias);

    int64_t k = weight.ndimension();
    int64_t dim = k - 2;

    auto stride = expand_dim_if_needed(stride_opt, "stride", dim);
    auto padding = expand_dim_if_needed(padding_opt, "padding", dim);
    auto dilation = expand_dim_if_needed(dilation_opt, "dilation", dim);
    auto output_padding = expand_dim_if_needed(output_padding_opt, "output_padding", dim);

    if (k == 3) {
        view1d_as_2d(stride, padding, dilation, output_padding);
        input = view4d(input);
        weight = view4d(weight);
    }

    at::Tensor output = transposed ?
                            acl_op::npu_convolution_transpose(input, weight, bias_opt, padding, output_padding, stride,
                                                              dilation, groups) :
                            acl_op::npu_convolution(input, weight, bias_opt, stride, padding, dilation, groups);

    if (k == 3) {
        output = view3d(output);
    }
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(const at::Tensor &input, const at::Tensor &grad,
                                                                        const at::Tensor &weight,
                                                                        at::IntArrayRef stride, at::IntArrayRef padding,
                                                                        at::IntArrayRef dilation, int64_t groups,
                                                                        std::array<bool, 3> grad_input_mask)
{
    int64_t dim = input.ndimension();

    std::tuple<at::Tensor, at::Tensor, at::Tensor> output;
    if (dim == 4) {
        output = acl_op::npu_conv2d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
    } else if (dim == 5) {
        output = acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
    }
    // Note:weight.grad should be equal weight
    if (std::get<1>(output).defined()) {
        std::get<1>(output) = at_npu::native::custom_ops::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
    }
    return output;
}


at::Tensor npu_convolution(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt,
                           at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups)
{
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value()) {
        if (bias_opt.value().defined()) {
            bias = bias_opt;
        }
    }

    int64_t dim = input.ndimension();
    auto kernel_size = weight.sizes().slice(2);

    at::Tensor output;
    if (dim == 4) {
        output = acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
    } else if (dim == 5) {
        bool is_dilated = false;
        for (int d : dilation) {
            is_dilated |= (d != 1);
        }
        output = (groups == 1 && !is_dilated) ?
                     at::slow_conv3d(input, weight, kernel_size, bias, stride, padding) :
                     acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
    }
    return output;
}

at::Tensor convolution_overrideable(const at::Tensor &input, const at::Tensor &weight,
                                    const c10::optional<at::Tensor> &bias_opt, c10::IntArrayRef stride,
                                    c10::IntArrayRef padding, c10::IntArrayRef dilation, bool transposed,
                                    c10::IntArrayRef output_padding, int64_t groups)
{
    int64_t dim = input.ndimension();
    auto kernel_size = weight.sizes().slice(2);

    at::Tensor output;
    if (dim == 4) {
        output = transposed ? acl_op::npu_conv_transpose2d(input, weight, bias_opt, padding, output_padding, stride,
                                                           dilation, groups) :
                              acl_op::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
    } else if (dim == 5) {
        bool is_dilated = false;
        for (int d : dilation) {
            is_dilated |= (d != 1);
        }
        output = (groups == 1 && !is_dilated) ?
                     at::slow_conv3d(input, weight, kernel_size, bias_opt, stride, padding) :
                     acl_op::npu_conv3d(input, weight, bias_opt, stride, padding, dilation, groups);
    }
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_overrideable(
    const at::Tensor &grad_output, const at::Tensor &input, const at::Tensor &weight, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, bool transposed, c10::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask)
{
    return acl_op::npu_convolution_backward(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor &grad_output_opt, const at::Tensor &input_opt, const at::Tensor &weight_opt,
    const c10::optional<at::IntArrayRef> bias_sizes_opt, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask)
{
    auto grad_output = grad_output_opt;
    auto input = input_opt;
    auto weight = weight_opt;
    op_plugin::utils::check_input_same_type_as_parameters(input, weight);

    auto k = weight.ndimension();
    int64_t dim = k - 2;

    TORCH_CHECK(dim > 0, "weight should have at least three dimensions" + OPS_ERROR(ErrCode::PARAM));

    auto &ctx = at::globalContext();
    ConvParams params;
    params.stride = expand_param_if_needed(stride, "stride", dim);
    params.padding = expand_param_if_needed(padding, "padding", dim);
    params.dilation = expand_param_if_needed(dilation, "dilation", dim);
    params.transposed = transposed;
    params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
    params.groups = groups;

    // Validate inputs.
    check_shape_backward(input, weight.sizes(), params);
    TORCH_CHECK(input.dim() == grad_output.dim(),
                "Expected input and grad_output to have the same number of dimensions, but got: ", input.dim(), " and ",
                grad_output.dim(), OPS_ERROR(ErrCode::PARAM));

    // output_padding is only supported for transposed convolutions
    if (!params.transposed) {
        for (auto pad : params.output_padding) {
            TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
                        params.output_padding, OPS_ERROR(ErrCode::VALUE));
        }
    }

    // Expand 1d -> 2d.
    // This is only done for backends that don't natively support 1d spatial input.
    if (k == 3) {
        // avoid accidentally going through NHWC for permuted 3d input.
        params.view1d_as_2d();
        grad_output = view4d(grad_output);
        input = view4d(input);
        weight = view4d(weight);
    }

    // Select appropriate backend to use.
    at::native::ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, true, params);

    // Call the backend.
    at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
    auto kernel_size = weight.sizes().slice(2);

    switch (backend) {
        case at::native::ConvBackend::Empty:
            if (output_mask[0]) {
                backend_grad_input = at::zeros_like(input);
            }
            if (output_mask[1]) {
                backend_grad_weight = at::zeros_like(weight);
            }
            if (output_mask[2]) {
                backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
            }
            break;
        case at::native::ConvBackend::Overrideable:
            // Only reach here when input is backend with out-of-source implementation.
            std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
                at::convolution_backward_overrideable(grad_output, input, weight, params.stride, params.padding,
                                                      params.dilation, params.transposed, params.output_padding,
                                                      params.groups, output_mask);
            break;
        case at::native::ConvBackend::Slow3d:
            std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_conv3d_backward(
                input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
            break;
        // Handle backends that don't natively support groups > 1.
        case at::native::ConvBackend::NnpackSpatial:
        case at::native::ConvBackend::Slow2d:
        case at::native::ConvBackend::SlowDilated2d:
        case at::native::ConvBackend::SlowDilated3d:
        case at::native::ConvBackend::SlowTranspose2d:
        case at::native::ConvBackend::SlowTranspose3d:
            {
                if (!params.transposed) {
                    std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
                        acl_op::npu_convolution_backward(input, grad_output, weight, params.stride, params.padding,
                                                         params.dilation, params.groups, output_mask);
                } else {
                    std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
                        acl_op::npu_convolution_transpose_backward(input, grad_output, weight, params.padding,
                                                                   params.output_padding, params.stride,
                                                                   params.dilation, params.groups, output_mask);
                }
                break;
            }
        // Backward is not supported for these backends.
        case at::native::ConvBackend::Winograd3x3Depthwise:
            TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd" + OPS_ERROR(ErrCode::NOT_SUPPORT));
            break;
        case at::native::ConvBackend::Xnnpack2d:
            TORCH_CHECK(false, "Backward is not supported for xnnpack" + OPS_ERROR(ErrCode::NOT_SUPPORT));
            break;
        default:
            TORCH_NPU_WARN_ONCE("Unkonwn Backward");
    }

    // Convert 2D inputs back to 1D for backends that don't natively support 1D
    // spatial inputs.
    if (output_mask[0]) {
        if (k == 3) {
            backend_grad_input = view3d(backend_grad_input);
        }
    }
    if (output_mask[1]) {
        if (k == 3) {
            backend_grad_weight = view3d(backend_grad_weight);
        }
    }
    if (output_mask[2]) {
        if (!backend_grad_bias.defined()) {
            // Calculate bias gradients outside of the backend for those that don't support it.
            backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
        }
    }

    return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

at::Tensor _slow_conv2d_forward(const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
                                at::IntArrayRef padding)
{
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor &bias = *bias_maybe_owned;
    at::Tensor output = acl_op::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
    return output;
}

at::Tensor &_slow_conv2d_forward_out(const at::Tensor &self, const at::Tensor &weight, at::IntArrayRef kernel_size,
                                     const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                                     at::IntArrayRef padding, at::Tensor &output)
{
    acl_op::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(const at::Tensor &grad_output,
                                                                     const at::Tensor &self, const at::Tensor &weight,
                                                                     at::IntArrayRef kernel_size,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding,
                                                                     std::array<bool, 3> output_mask)
{
    return acl_op::npu_convolution_backward(self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
void check_shape_forward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const at::Tensor& bias,
    const ConvParams& params) {
  int64_t k = input.ndimension();
  TORCH_CHECK(k >= 2, "The length of input_size should be at least 2" + OPS_ERROR(ErrCode::PARAM));

  int64_t weight_dim = static_cast<int64_t>(weight_sizes.size());
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));

  TORCH_CHECK(weight_dim == k,
      "Expected ", weight_dim, "-dimensional input for ", weight_dim,
      "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
      input.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] >= groups,
      "Given groups=", groups, ", expected weight to be at least ", groups,
      " at dimension 0, but got weight of size ", weight_sizes, " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] % groups == 0,
      "Given groups=", groups, ", expected weight to be divisible by ",
      groups, " at dimension 0, but got weight of size [", weight_sizes,
      "] instead", OPS_ERROR(ErrCode::PARAM));

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
        "Given groups=", groups, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ",
        (weight_sizes[1] * groups), " channels, but got ", input.size(1),
        " channels instead", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(k >= 3, "When transposed is true, the length of input_size should be at least 3" + OPS_ERROR(ErrCode::PARAM));
    int64_t dim_del = k - 2;
    TORCH_CHECK(padding.size() >= dim_del, "The length of padding should be at least ", dim_del, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= dim_del, "The length of dilation should be at least ", dim_del, OPS_ERROR(ErrCode::PARAM));

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel" + OPS_ERROR(ErrCode::PARAM));

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (size_t i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). Kernel size: (",
                  kernel_ss.str(), "). Kernel size can't be greater than actual input size", OPS_ERROR(ErrCode::PARAM));
    }
  } else {
    // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ", weight_sizes[0],
        " channels, but got ", input.size(1), " channels instead", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  }
}

void check_shape_backward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const ConvParams& params) {
  check_shape_forward(input, weight_sizes, at::Tensor(), params);
}

at::native::ConvBackend select_conv_backend(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const bool need_backward,
    const ConvParams& params) {
  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    return at::native::ConvBackend::Empty;
  } else if (input.numel() == 0) {
    TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes(), OPS_ERROR(ErrCode::PARAM));
  }

  if (torch_npu::utils::is_npu(input)) {
    // backends without support for groups
    if (params.transposed) {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::SlowTranspose2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::SlowTranspose3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    } else {  /* Not transposed */
      if (input.ndimension() == 4) {
        if (params.is_dilated()) {
          return at::native::ConvBackend::SlowDilated2d;
        } else {
          return at::native::ConvBackend::Slow2d;
        }
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::Slow3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    return at::native::ConvBackend::Overrideable;
  }
  // Error out if no suitable backend was found.
    TORCH_CHECK(false, "unsupported ConvNd parameters" + OPS_ERROR(ErrCode::NOT_SUPPORT));
}

// Selects a backend for convolution based on the inputs and params.
at::native::ConvBackend select_conv_backend(
    const at::Tensor& input_r,
    const at::Tensor& weight_r,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_opt,
    at::IntArrayRef padding_opt,
    at::IntArrayRef dilation_opt,
    bool transposed,
    at::IntArrayRef output_padding_opt,
    int64_t groups) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  auto& ctx = at::globalContext();
  auto k = weight_r.ndimension();
  int64_t dim = k - 2;
  ConvParams params;
  params.stride = expand_param_if_needed(stride_opt, "stride", dim);
  params.padding = expand_param_if_needed(padding_opt, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_opt, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = expand_param_if_needed(output_padding_opt, "output_padding", dim);
  params.groups = groups;

  auto input = input_r;
  auto weight = weight_r;
  check_shape_forward(input, weight.sizes(), bias, params);

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3 && !input.is_mkldnn()) {
    // avoid accidentally going through NHWC for permuted 3d input.
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  auto bias_sizes_opt = bias.defined() ? c10::optional<at::IntArrayRef>(bias.sizes()) : c10::nullopt;
  bool need_backward = c10::GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  return select_conv_backend(input, weight, bias_sizes_opt, need_backward, params);
}

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  return at::convolution(input, weight, bias, stride, padding, dilation, true, output_padding, groups);
}

at::Tensor conv_transpose3d(
    const at::Tensor& input_opt,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  at::Tensor input;
  bool is_batched;
  std::tie(input, is_batched) = batchify(input_opt, 3, "conv_transpose3d");
  auto output = at::convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  return is_batched ? output : output.squeeze(0);
}

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  return at::_convolution(
      input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false, false, false);
}

at::Tensor _convolution(
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_opt,
    at::IntArrayRef padding_opt,
    at::IntArrayRef dilation_opt,
    bool transposed,
    at::IntArrayRef output_padding_opt,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  at::Tensor input = input_opt;
  at::Tensor weight = weight_opt;

  const at::Tensor& bias_val = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  at::Tensor bias = bias_val;
  op_plugin::utils::check_input_same_type_as_parameters(input, weight, bias);

  int64_t k = weight.ndimension();
  int64_t dim = k - 2;

  auto stride = expand_dim_if_needed(stride_opt, "stride", dim);
  auto padding = expand_dim_if_needed(padding_opt, "padding", dim);
  auto dilation = expand_dim_if_needed(dilation_opt, "dilation", dim);
  auto output_padding = expand_dim_if_needed(output_padding_opt, "output_padding", dim);

  if (k == 3) {
    view1d_as_2d(stride, padding, dilation, output_padding);
    input = view4d(input);
    weight = view4d(weight);
  }

  at::Tensor output = transposed ? acl_op::npu_convolution_transpose(
      input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
      acl_op::npu_convolution(input, weight, bias_opt, stride, padding, dilation, groups);

  if (k == 3) {
    output = view3d(output);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  std::tuple<at::Tensor, at::Tensor, at::Tensor> output;
  if (dim == 4) {
    output = acl_op::npu_conv2d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
  } else if (dim == 5) {
    output = acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
  }
  // Note:weight.grad should be equal weight
  if (std::get<1>(output).defined()) {
    std::get<1>(output) = at_npu::native::custom_ops::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
  }
  return output;
}


at::Tensor npu_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::optional<at::Tensor> bias = c10::nullopt;
  if (bias_opt.has_value()) {
    if (bias_opt.value().defined()) {
      bias = bias_opt;
    }
  }

  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    output = acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
  } else if (dim == 5) {
    bool is_dilated = false;
    for (int d : dilation) {
      is_dilated |= (d != 1);
    }
    output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias, stride, padding) :
        acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return output;
}

at::Tensor convolution_overrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups) {
  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    output = transposed ?
        acl_op::npu_conv_transpose2d(input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
        acl_op::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
  } else if (dim == 5) {
    bool is_dilated = false;
    for (int d : dilation) {
      is_dilated |= (d != 1);
    }
    output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias_opt, stride, padding) :
       acl_op::npu_conv3d(input, weight, bias_opt, stride, padding, dilation, groups);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_overrideable(
    const at::Tensor & grad_output,
    const at::Tensor & input,
    const at::Tensor & weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  return acl_op::npu_convolution_backward(
      input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output_opt,
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_opt;
  auto input = input_opt;
  auto weight = weight_opt;
  op_plugin::utils::check_input_same_type_as_parameters(input, weight);

  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions" + OPS_ERROR(ErrCode::PARAM));

  auto& ctx = at::globalContext();
  ConvParams params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(), " and ", grad_output.dim(), OPS_ERROR(ErrCode::PARAM));

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
          params.output_padding, OPS_ERROR(ErrCode::PARAM));
    }
  }

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3) {
    // avoid accidentally going through NHWC for permuted 3d input.
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  at::native::ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, true, params);

  // Call the backend.
  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  auto kernel_size = weight.sizes().slice(2);

  switch(backend) {
    case at::native::ConvBackend::Empty:
      if (output_mask[0]) {
        backend_grad_input = at::zeros_like(input);
      }
      if (output_mask[1]) {
        backend_grad_weight = at::zeros_like(weight);
      }
      if (output_mask[2]) {
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
      }
      break;
    case at::native::ConvBackend::Overrideable:
      // Only reach here when input is backend with out-of-source implementation.
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = at::convolution_backward_overrideable(
          grad_output, input, weight, params.stride, params.padding, params.dilation, params.transposed,
          params.output_padding, params.groups, output_mask);
      break;
    case at::native::ConvBackend::Slow3d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_conv3d_backward(
          input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
      break;
    // Handle backends that don't natively support groups > 1.
    case at::native::ConvBackend::NnpackSpatial:
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowDilated3d:
    case at::native::ConvBackend::SlowTranspose2d:
    case at::native::ConvBackend::SlowTranspose3d: {
      if (!params.transposed) {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_backward(
            input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
      } else {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_transpose_backward(
            input, grad_output, weight, params.padding, params.output_padding, params.stride,
            params.dilation, params.groups, output_mask);
      }
      break;
    }
    // Backward is not supported for these backends.
    case at::native::ConvBackend::Winograd3x3Depthwise:
      TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd" + OPS_ERROR(ErrCode::NOT_SUPPORT));
      break;
    case at::native::ConvBackend::Xnnpack2d:
      TORCH_CHECK(false, "Backward is not supported for xnnpack" + OPS_ERROR(ErrCode::NOT_SUPPORT));
      break;
    default:
        TORCH_NPU_WARN_ONCE("Unkonwn Backward");
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0]) {
    if (k == 3) {
      backend_grad_input = view3d(backend_grad_input);
    }
  }
  if (output_mask[1]) {
    if (k == 3) {
      backend_grad_weight = view3d(backend_grad_weight);
    }
  }
  if (output_mask[2]) {
    if (!backend_grad_bias.defined()) {
      // Calculate bias gradients outside of the backend for those that don't support it.
      backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
    }
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

at::Tensor _slow_conv2d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  at::Tensor output = acl_op::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return output;
}

at::Tensor& _slow_conv2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  acl_op::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  return acl_op::npu_convolution_backward(self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}

#endif
#if VERSION_BETWEEN(V2R1, V2R1)
void check_shape_forward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const at::Tensor& bias,
    const ConvParams& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = static_cast<int64_t>(weight_sizes.size());
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));

  TORCH_CHECK(weight_dim == k,
      "Expected ", weight_dim, "-dimensional input for ", weight_dim,
      "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
      input.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] >= groups,
      "Given groups=", groups, ", expected weight to be at least ", groups,
      " at dimension 0, but got weight of size ", weight_sizes, " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] % groups == 0,
      "Given groups=", groups, ", expected weight to be divisible by ",
      groups, " at dimension 0, but got weight of size [", weight_sizes,
      "] instead", OPS_ERROR(ErrCode::PARAM));

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
        "Given groups=", groups, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ",
        (weight_sizes[1] * groups), " channels, but got ", input.size(1),
        " channels instead", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel", OPS_ERROR(ErrCode::PARAM));

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (uint64_t i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). Kernel size: (",
                  kernel_ss.str(), "). Kernel size can't be greater than actual input size", OPS_ERROR(ErrCode::PARAM));
    }
  } else {
    // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ", weight_sizes[0],
        " channels, but got ", input.size(1), " channels instead", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  }
}

void check_shape_backward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const ConvParams& params) {
  check_shape_forward(input, weight_sizes, at::Tensor(), params);
}

at::native::ConvBackend select_conv_backend(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const bool need_backward,
    const ConvParams& params) {
  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    return at::native::ConvBackend::Empty;
  } else if (input.numel() == 0) {
    TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes(), OPS_ERROR(ErrCode::NOT_SUPPORT));
  }

  if (torch_npu::utils::is_npu(input)) {
    // backends without support for groups
    if (params.transposed) {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::SlowTranspose2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::SlowTranspose3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    } else {  /* Not transposed */
      if (input.ndimension() == 4) {
        if (params.is_dilated()) {
          return at::native::ConvBackend::SlowDilated2d;
        } else {
          return at::native::ConvBackend::Slow2d;
        }
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::Slow3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported"+ OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    return at::native::ConvBackend::Overrideable;
  }
  // Error out if no suitable backend was found.
    TORCH_CHECK(false, "unsupported ConvNd parameters"+ OPS_ERROR(ErrCode::NOT_SUPPORT));
}

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  return at::convolution(input, weight, bias, stride, padding, dilation, true, output_padding, groups);
}

at::Tensor conv_transpose3d(
    const at::Tensor& input_opt,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  at::Tensor input;
  bool is_batched;
  std::tie(input, is_batched) = batchify(input_opt, 3, "conv_transpose3d");
  auto output = at::convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  return is_batched ? output : output.squeeze(0);
}

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  return at::_convolution(
      input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false, false, false);
}

at::Tensor _convolution(
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_opt,
    at::IntArrayRef padding_opt,
    at::IntArrayRef dilation_opt,
    bool transposed,
    at::IntArrayRef output_padding_opt,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  at::Tensor input = input_opt;
  at::Tensor weight = weight_opt;

  const at::Tensor& bias_val = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  at::Tensor bias = bias_val;
  op_plugin::utils::check_input_same_type_as_parameters(input, weight, bias);

  int64_t k = weight.ndimension();
  int64_t dim = k - 2;

  auto stride = expand_dim_if_needed(stride_opt, "stride", dim);
  auto padding = expand_dim_if_needed(padding_opt, "padding", dim);
  auto dilation = expand_dim_if_needed(dilation_opt, "dilation", dim);
  auto output_padding = expand_dim_if_needed(output_padding_opt, "output_padding", dim);

  if (k == 3) {
    view1d_as_2d(stride, padding, dilation, output_padding);
    input = view4d(input);
    weight = view4d(weight);
  }

  at::Tensor output = transposed ? acl_op::npu_convolution_transpose(
      input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
      acl_op::npu_convolution(input, weight, bias_opt, stride, padding, dilation, groups);

  if (k == 3) {
    output = view3d(output);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  std::tuple<at::Tensor, at::Tensor, at::Tensor> output;
  if (dim == 4) {
    output = acl_op::npu_conv2d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
  } else if (dim == 5) {
    output = acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
  }
  // Note:weight.grad should be equal weight
  if (std::get<1>(output).defined()) {
    std::get<1>(output) = at_npu::native::custom_ops::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
  }
  return output;
}


at::Tensor npu_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::optional<at::Tensor> bias = c10::nullopt;
  if (bias_opt.has_value()) {
    if (bias_opt.value().defined()) {
      bias = bias_opt;
    }
  }

  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    output = acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
  } else if (dim == 5) {
    bool is_dilated = false;
    for (int d : dilation) {
      is_dilated |= (d != 1);
    }
    output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias, stride, padding) :
        acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
  return output;
}

at::Tensor convolution_overrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups) {
  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    output = transposed ?
        acl_op::npu_conv_transpose2d(input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
        acl_op::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
  } else if (dim == 5) {
    bool is_dilated = false;
    for (int d : dilation) {
      is_dilated |= (d != 1);
    }
    output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias_opt, stride, padding) :
       acl_op::npu_conv3d(input, weight, bias_opt, stride, padding, dilation, groups);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_overrideable(
    const at::Tensor & grad_output,
    const at::Tensor & input,
    const at::Tensor & weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  return acl_op::npu_convolution_backward(
      input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output_opt,
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_opt;
  auto input = input_opt;
  auto weight = weight_opt;
  op_plugin::utils::check_input_same_type_as_parameters(input, weight);

  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions" + OPS_ERROR(ErrCode::PARAM));

  auto& ctx = at::globalContext();
  ConvParams params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(), " and ", grad_output.dim(), OPS_ERROR(ErrCode::PARAM));

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
          params.output_padding, OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
  }

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3) {
    // avoid accidentally going through NHWC for permuted 3d input.
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  at::native::ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, true, params);

  // Call the backend.
  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  auto kernel_size = weight.sizes().slice(2);

  switch(backend) {
    case at::native::ConvBackend::Empty:
      if (output_mask[0]) {
        backend_grad_input = at::zeros_like(input);
      }
      if (output_mask[1]) {
        backend_grad_weight = at::zeros_like(weight);
      }
      if (output_mask[2]) {
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
      }
      break;
    case at::native::ConvBackend::Overrideable:
      // Only reach here when input is backend with out-of-source implementation.
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = at::convolution_backward_overrideable(
          grad_output, input, weight, params.stride, params.padding, params.dilation, params.transposed,
          params.output_padding, params.groups, output_mask);
      break;
    case at::native::ConvBackend::Slow3d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_conv3d_backward(
          input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
      break;
    // Handle backends that don't natively support groups > 1.
    case at::native::ConvBackend::NnpackSpatial:
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowDilated3d:
    case at::native::ConvBackend::SlowTranspose2d:
    case at::native::ConvBackend::SlowTranspose3d: {
      if (!params.transposed) {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_backward(
            input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
      } else {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_transpose_backward(
            input, grad_output, weight, params.padding, params.output_padding, params.stride,
            params.dilation, params.groups, output_mask);
      }
      break;
    }
    // Backward is not supported for these backends.
    case at::native::ConvBackend::Winograd3x3Depthwise:
      TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd", OPS_ERROR(ErrCode::NOT_SUPPORT));
      break;
    case at::native::ConvBackend::Xnnpack2d:
      TORCH_CHECK(false, "Backward is not supported for xnnpack", OPS_ERROR(ErrCode::NOT_SUPPORT));
      break;
    default:
        TORCH_NPU_WARN_ONCE("Unkonwn Backward");
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0]) {
    if (k == 3) {
      backend_grad_input = view3d(backend_grad_input);
    }
  }
  if (output_mask[1]) {
    if (k == 3) {
      backend_grad_weight = view3d(backend_grad_weight);
    }
  }
  if (output_mask[2]) {
    if (!backend_grad_bias.defined()) {
      // Calculate bias gradients outside of the backend for those that don't support it.
      backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
    }
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

at::Tensor _slow_conv2d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  at::Tensor output = acl_op::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return output;
}

at::Tensor& _slow_conv2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  acl_op::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  return acl_op::npu_convolution_backward(self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
void check_shape_forward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const at::Tensor& bias,
    const ConvParams& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = static_cast<int64_t>(weight_sizes.size());
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported" + OPS_ERROR(ErrCode::NOT_SUPPORT));

  TORCH_CHECK(weight_dim == k,
      "Expected ", weight_dim, "-dimensional input for ", weight_dim,
      "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
      input.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] >= groups,
      "Given groups=", groups, ", expected weight to be at least ", groups,
      " at dimension 0, but got weight of size ", weight_sizes, " instead", OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(weight_sizes[0] % groups == 0,
      "Given groups=", groups, ", expected weight to be divisible by ",
      groups, " at dimension 0, but got weight of size [", weight_sizes,
      "] instead", OPS_ERROR(ErrCode::PARAM));

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
        "Given groups=", groups, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ",
        (weight_sizes[1] * groups), " channels, but got ", input.size(1),
        " channels instead", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel", OPS_ERROR(ErrCode::PARAM));

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (uint64_t i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). Kernel size: (",
                  kernel_ss.str(), "). Kernel size can't be greater than actual input size", OPS_ERROR(ErrCode::PARAM));
    }
  } else {
    // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ", weight_sizes[0],
        " channels, but got ", input.size(1), " channels instead", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
        ", but got bias of size ", bias.sizes(), " instead", OPS_ERROR(ErrCode::PARAM));
  }
}

void check_shape_backward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const ConvParams& params) {
  check_shape_forward(input, weight_sizes, at::Tensor(), params);
}

at::native::ConvBackend select_conv_backend(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const bool need_backward,
    const ConvParams& params) {
  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    return at::native::ConvBackend::Empty;
  } else if (input.numel() == 0) {
    TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes(), OPS_ERROR(ErrCode::NOT_SUPPORT));
  }

  if (torch_npu::utils::is_npu(input)) {
    // backends without support for groups
    if (params.transposed) {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::SlowTranspose2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::SlowTranspose3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported", OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    } else {  /* Not transposed */
      if (input.ndimension() == 4) {
        if (params.is_dilated()) {
          return at::native::ConvBackend::SlowDilated2d;
        } else {
          return at::native::ConvBackend::Slow2d;
        }
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::Slow3d;
      } else {
        TORCH_CHECK(false, "Only 4D or 5D input is supported", OPS_ERROR(ErrCode::NOT_SUPPORT));
      }
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    return at::native::ConvBackend::Overrideable;
  }
    // Error out if no suitable backend was found.
    TORCH_CHECK(false, "unsupported ConvNd parameters", OPS_ERROR(ErrCode::NOT_SUPPORT));
}

// Selects a backend for convolution based on the inputs and params.
at::native::ConvBackend select_conv_backend(
    const at::Tensor& input_r,
    const at::Tensor& weight_r,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_opt,
    at::IntArrayRef padding_opt,
    at::IntArrayRef dilation_opt,
    bool transposed,
    at::IntArrayRef output_padding_opt,
    int64_t groups) {
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor& bias = *bias_maybe_owned;

    auto& ctx = at::globalContext();
    auto k = weight_r.ndimension();
    int64_t dim = k - 2;
    ConvParams params;
    params.stride = expand_param_if_needed(stride_opt, "stride", dim);
    params.padding = expand_param_if_needed(padding_opt, "padding", dim);
    params.dilation = expand_param_if_needed(dilation_opt, "dilation", dim);
    params.transposed = transposed;
    params.output_padding = expand_param_if_needed(output_padding_opt, "output_padding", dim);
    params.groups = groups;

    auto input = input_r;
    auto weight = weight_r;
    check_shape_forward(input, weight.sizes(), bias, params);

    // Expand 1d -> 2d.
    // This is only done for backends that don't natively support 1d spatial input.
    if (k == 3 && !input.is_mkldnn()) {
        // avoid accidentally going through NHWC for permuted 3d input.
        params.view1d_as_2d();
        input = view4d(input);
        weight = view4d(weight);
    }

    auto bias_sizes_opt = bias.defined() ? c10::optional<at::IntArrayRef>(bias.sizes()) : c10::nullopt;
    bool need_backward = c10::GradMode::is_enabled() &&
        (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
    return select_conv_backend(input, weight, bias_sizes_opt, need_backward, params);
}

at::Tensor conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
    return at::convolution(input, weight, bias, stride, padding, dilation, true, output_padding, groups);
}

at::Tensor conv_transpose3d(
    const at::Tensor& input_opt,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor& bias = *bias_maybe_owned;

    at::Tensor input;
    bool is_batched;
    std::tie(input, is_batched) = batchify(input_opt, 3, "conv_transpose3d");
    auto output = at::convolution(
        input, weight, bias, stride, padding, dilation, true, output_padding, groups);
    return is_batched ? output : output.squeeze(0);
}

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
    return at::_convolution(
        input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false, false, false);
}

at::Tensor _convolution(
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_opt,
    at::IntArrayRef padding_opt,
    at::IntArrayRef dilation_opt,
    bool transposed,
    at::IntArrayRef output_padding_opt,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
    at::Tensor input = input_opt;
    at::Tensor weight = weight_opt;

    const at::Tensor& bias_val = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    at::Tensor bias = bias_val;
    op_plugin::utils::check_input_same_type_as_parameters(input, weight, bias);

    int64_t k = weight.ndimension();
    int64_t dim = k - 2;

    auto stride = expand_dim_if_needed(stride_opt, "stride", dim);
    auto padding = expand_dim_if_needed(padding_opt, "padding", dim);
    auto dilation = expand_dim_if_needed(dilation_opt, "dilation", dim);
    auto output_padding = expand_dim_if_needed(output_padding_opt, "output_padding", dim);

    if (k == 3) {
        view1d_as_2d(stride, padding, dilation, output_padding);
        input = view4d(input);
        weight = view4d(weight);
    }

    at::Tensor output = transposed ? acl_op::npu_convolution_transpose(
        input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
        acl_op::npu_convolution(input, weight, bias_opt, stride, padding, dilation, groups);

    if (k == 3) {
        output = view3d(output);
    }
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
    int64_t dim = input.ndimension();

    std::tuple<at::Tensor, at::Tensor, at::Tensor> output;
    if (dim == 4) {
        output = acl_op::npu_conv2d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
    } else if (dim == 5) {
        output = acl_op::npu_conv3d_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
    }
    // Note:weight.grad should be equal weight
    if (std::get<1>(output).defined()) {
        std::get<1>(output) = at_npu::native::custom_ops::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
    }
    return output;
}

at::Tensor npu_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value()) {
        if (bias_opt.value().defined()) {
            bias = bias_opt;
        }
    }

    int64_t dim = input.ndimension();
    auto kernel_size = weight.sizes().slice(2);

    at::Tensor output;
    if (dim == 4) {
        output = acl_op::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
    } else if (dim == 5) {
        bool is_dilated = false;
        for (int d : dilation) {
            is_dilated |= (d != 1);
        }
        output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias, stride, padding) :
            acl_op::npu_conv3d(input, weight, bias, stride, padding, dilation, groups);
    }
    return output;
}

at::Tensor convolution_overrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups) {
    int64_t dim = input.ndimension();
    auto kernel_size = weight.sizes().slice(2);

    at::Tensor output;
    if (dim == 4) {
        output = transposed ?
            acl_op::npu_conv_transpose2d(input, weight, bias_opt, padding, output_padding, stride, dilation, groups) :
            acl_op::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
    } else if (dim == 5) {
        bool is_dilated = false;
        for (int d : dilation) {
            is_dilated |= (d != 1);
        }
        output = (groups == 1 && !is_dilated) ? at::slow_conv3d(input, weight, kernel_size, bias_opt, stride, padding) :
          acl_op::npu_conv3d(input, weight, bias_opt, stride, padding, dilation, groups);
    }
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_overrideable(
    const at::Tensor & grad_output,
    const at::Tensor & input,
    const at::Tensor & weight,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
    return acl_op::npu_convolution_backward(
        input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output_opt,
    const at::Tensor& input_opt,
    const at::Tensor& weight_opt,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
    auto grad_output = grad_output_opt;
    auto input = input_opt;
    auto weight = weight_opt;
    op_plugin::utils::check_input_same_type_as_parameters(input, weight);

    auto k = weight.ndimension();
    int64_t dim = k - 2;

    TORCH_CHECK(dim > 0, "weight should have at least three dimensions" + OPS_ERROR(ErrCode::PARAM));

    auto& ctx = at::globalContext();
    ConvParams params;
    params.stride = expand_param_if_needed(stride, "stride", dim);
    params.padding = expand_param_if_needed(padding, "padding", dim);
    params.dilation = expand_param_if_needed(dilation, "dilation", dim);
    params.transposed = transposed;
    params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
    params.groups = groups;

    // Validate inputs.
    check_shape_backward(input, weight.sizes(), params);
    TORCH_CHECK(input.dim() == grad_output.dim(),
        "Expected input and grad_output to have the same number of dimensions, but got: ",
        input.dim(), " and ", grad_output.dim(), OPS_ERROR(ErrCode::PARAM));

    // output_padding is only supported for transposed convolutions
    if (!params.transposed) {
        for (auto pad : params.output_padding) {
            TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
                params.output_padding, OPS_ERROR(ErrCode::PARAM));
        }
    }

    // Expand 1d -> 2d.
    // This is only done for backends that don't natively support 1d spatial input.
    if (k == 3) {
        // avoid accidentally going through NHWC for permuted 3d input.
        params.view1d_as_2d();
        grad_output = view4d(grad_output);
        input = view4d(input);
        weight = view4d(weight);
    }

    // Select appropriate backend to use.
    at::native::ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, true, params);

    // Call the backend.
    at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
    auto kernel_size = weight.sizes().slice(2);

    switch(backend) {
        case at::native::ConvBackend::Empty:
            if (output_mask[0]) {
                backend_grad_input = at::zeros_like(input);
            }
            if (output_mask[1]) {
                backend_grad_weight = at::zeros_like(weight);
            }
            if (output_mask[2]) {
                backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
            }
            break;
        case at::native::ConvBackend::Overrideable:
            // Only reach here when input is backend with out-of-source implementation.
            std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = at::convolution_backward_overrideable(
                grad_output, input, weight, params.stride, params.padding, params.dilation, params.transposed,
                params.output_padding, params.groups, output_mask);
            break;
        case at::native::ConvBackend::Slow3d:
            std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_conv3d_backward(
                input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
            break;
        // Handle backends that don't natively support groups > 1.
        case at::native::ConvBackend::NnpackSpatial:
        case at::native::ConvBackend::Slow2d:
        case at::native::ConvBackend::SlowDilated2d:
        case at::native::ConvBackend::SlowDilated3d:
        case at::native::ConvBackend::SlowTranspose2d:
        case at::native::ConvBackend::SlowTranspose3d: {
            if (!params.transposed) {
                std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_backward(
                    input, grad_output, weight, params.stride, params.padding, params.dilation, params.groups, output_mask);
            } else {
                std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = acl_op::npu_convolution_transpose_backward(
                    input, grad_output, weight, params.padding, params.output_padding, params.stride,
                    params.dilation, params.groups, output_mask);
            }
            break;
        }
        // Backward is not supported for these backends.
        case at::native::ConvBackend::Winograd3x3Depthwise:
            TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd" + OPS_ERROR(ErrCode::NOT_SUPPORT));
            break;
        case at::native::ConvBackend::Xnnpack2d:
            TORCH_CHECK(false, "Backward is not supported for xnnpack" + OPS_ERROR(ErrCode::NOT_SUPPORT));
            break;
        default:
            TORCH_NPU_WARN_ONCE("Unkonwn Backward");
    }

    // Convert 2D inputs back to 1D for backends that don't natively support 1D
    // spatial inputs.
    if (output_mask[0]) {
        if (k == 3) {
            backend_grad_input = view3d(backend_grad_input);
        }
    }
    if (output_mask[1]) {
        if (k == 3) {
            backend_grad_weight = view3d(backend_grad_weight);
        }
    }
    if (output_mask[2]) {
        if (!backend_grad_bias.defined()) {
            // Calculate bias gradients outside of the backend for those that don't support it.
            backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
        }
    }

    return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

at::Tensor _slow_conv2d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
    c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const at::Tensor& bias = *bias_maybe_owned;
    at::Tensor output = acl_op::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
    return output;
}

at::Tensor& _slow_conv2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
    acl_op::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _slow_conv2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    std::array<bool, 3> output_mask) {
    return acl_op::npu_convolution_backward(self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}
#endif
} // namespace acl_op
