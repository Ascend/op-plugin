#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/custom_functions/BackwardManual.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace op_plugin {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_transpose_backward(const at::Tensor & input,
 const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding,
 at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> grad_input_mask) {
    return acl_op::npu_convolution_transpose_backward(input, grad, weight, padding, output_padding,
  stride, dilation, groups, grad_input_mask);
}

at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor & y_grad, const at::Tensor & y,
const at::Tensor & mask, at::Scalar scale, bool fixed_triu_mask){
    return acl_op::npu_scaled_masked_softmax_backward( y_grad, y, mask, scale,fixed_triu_mask);
}

at::Tensor npu_dtype_cast_backward(const at::Tensor& grad, at::ScalarType dtype){
    return acl_op::npu_dtype_cast_backward(grad, dtype);
}

at::Tensor npu_binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction){
    return acl_op::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction);
}

} //op_plugin
