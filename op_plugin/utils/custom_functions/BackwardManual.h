#ifndef __PULGIN_UTILS_CUSTOM_MANUAL_BACKWARD__
#define __PULGIN_UTILS_CUSTOM_MANUAL_BACKWARD__

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace op_plugin {
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_transpose_backward(const at::Tensor & input,
const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> grad_input_mask);

at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y,
const at::Tensor & mask, at::Scalar scale, bool fixed_triu_mask);

at::Tensor npu_dtype_cast_backward(const at::Tensor& grad, at::ScalarType dtype);

at::Tensor npu_binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction);
} //op_plugin

#endif
