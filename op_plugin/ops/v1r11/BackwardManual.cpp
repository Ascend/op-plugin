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
#include "op_plugin/ops/v1r11/BackwardManual.h"
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

at::Tensor npu_confusion_transpose_backward(const at::Tensor& grad, at::IntArrayRef perm, at::IntArrayRef shape,
                                            bool transpose_first) {
  return acl_op::npu_confusion_transpose_backward(grad, perm, shape, transpose_first);
}

at::Tensor npu_max_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, at::IntArrayRef sizes,
                            bool keepdim) {
  return acl_op::npu_max_backward(grad, dim, indices, sizes, keepdim);
}

at::Tensor npu_min_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, at::IntArrayRef sizes,
                            bool keepdim) {
  return acl_op::npu_min_backward(grad, dim, indices, sizes, keepdim);
}

at::Tensor npu_ps_roi_pooling_backward(const at::Tensor& output_grad, const at::Tensor& rois, double spatial_scale,
                                       int64_t group_size, int64_t output_dim, at::IntArrayRef input_size) {
  return acl_op::npu_ps_roi_pooling_backward(output_grad, rois, spatial_scale, group_size, output_dim, input_size);
}

at::Tensor npu_bmm_v2_mat1_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    at::IntArrayRef size) {
  return acl_op::npu_bmm_v2_mat1_backward(grad, mat1, mat2, size);
}

at::Tensor npu_bmm_v2_mat2_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    at::IntArrayRef size) {
  return acl_op::npu_bmm_v2_mat2_backward(grad, mat1, mat2, size);
}

at::Tensor celu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Tensor& output) {
  return acl_op::celu_backward(grad_output, alpha, output);
}

at::Tensor elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                        const at::Scalar& input_scale, const at::Tensor& output) {
  return acl_op::elu_backward(grad_output, alpha, scale, input_scale, output);
}

at::Tensor selu_backward(const at::Tensor& grad_output, const at::Tensor& result) {
  return acl_op::selu_backward(grad_output, result);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt, const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt, const at::Tensor& input, const at::Tensor& batch_sizes,
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& init_h, const at::Tensor& init_c,
    const at::Tensor& y, const at::Tensor& h, const at::Tensor& c, const at::Tensor& i, const at::Tensor& j,
    const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc, bool flag_direction) {
  return acl_op::npu_lstm_data_backward(grady_opt, gradh_opt, gradc_opt, input, batch_sizes, weight, bias, init_h,
                                        init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}
}  // namespace op_plugin
