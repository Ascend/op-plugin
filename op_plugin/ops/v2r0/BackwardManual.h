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

#ifndef __OP_PULGIN_OPS_V2R0_BACKWARD_MANUAL__
#define __OP_PULGIN_OPS_V2R0_BACKWARD_MANUAL__

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

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

at::Tensor npu_confusion_transpose_backward(const at::Tensor& grad, at::IntArrayRef perm, c10::SymIntArrayRef shape,
                                            bool transpose_first);
at::Tensor npu_max_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, c10::SymIntArrayRef sizes,
                            bool keepdim);
at::Tensor npu_min_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, c10::SymIntArrayRef sizes,
                            bool keepdim);
at::Tensor npu_ps_roi_pooling_backward(const at::Tensor& output_grad, const at::Tensor& rois, double spatial_scale,
                                       int64_t group_size, int64_t output_dim, c10::SymIntArrayRef input_size);
at::Tensor npu_bmm_v2_mat1_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    c10::SymIntArrayRef size);
at::Tensor npu_bmm_v2_mat2_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    c10::SymIntArrayRef size);
at::Tensor celu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Tensor& output);
at::Tensor elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                        const at::Scalar& input_scale, const at::Tensor& output);
at::Tensor selu_backward(const at::Tensor& grad_output, const at::Tensor& result);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt, const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt, const at::Tensor& input, const at::Tensor& batch_sizes,
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& init_h, const at::Tensor& init_c,
    const at::Tensor& y, const at::Tensor& h, const at::Tensor& c, const at::Tensor& i, const at::Tensor& j,
    const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc, bool flag_direction);
at::Tensor l1_loss_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
                            int64_t reduction);
at::Tensor kl_div_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, bool log_target);
}  // namespace op_plugin
#endif
