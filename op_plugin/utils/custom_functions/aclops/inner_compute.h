// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE__
#define __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE__

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

namespace acl_op {
at::Tensor npu_confusion_transpose_backward(const at::Tensor& grad, at::IntArrayRef perm, at::IntArrayRef shape,
                                            bool transpose_first);
at::Tensor npu_max_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, at::IntArrayRef sizes,
                            bool keepdim);
at::Tensor npu_min_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, at::IntArrayRef sizes,
                            bool keepdim);
at::Tensor npu_ps_roi_pooling_backward(const at::Tensor& output_grad, const at::Tensor& rois, double spatial_scale,
                                       int64_t group_size, int64_t output_dim, at::IntArrayRef input_size);
at::Tensor npu_bmm_v2_mat1_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    at::IntArrayRef size);
at::Tensor npu_bmm_v2_mat2_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,
                                    at::IntArrayRef size);
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
at::Tensor embedding_common_nocheck(const at::Tensor& weight, const at::Tensor& indices);
at::Tensor gelu_common_nocheck(const at::Tensor& self);
at::Tensor gelu_backward_common_nocheck(const at::Tensor& grad, const at::Tensor& self);
std::tuple<at::Tensor, at::Tensor> grid_sampler3d_backward_common_nocheck(const at::Tensor& grad, const at::Tensor& input,
                                                                          const at::Tensor& grid, int64_t interpolation_mode,
                                                                          int64_t padding_mode, bool align_corners);
at::Tensor& sum_out_common_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype);
at::Tensor sum_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_expand_outplace(const at::Tensor& to_expand1, const at::Tensor& to_expand2,
                                                                   const at::Tensor& to_expand3, const char* api_name);
at::Tensor& where_out_nocheck(at::Tensor& out, const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other);
void index_copy_npu_par_check(const int64_t dim, const at::Tensor& index, const at::Tensor& source, const at::Tensor& result);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_svd_out_common(const at::Tensor& A, const bool full_matrices,
                                                                        const bool compute_uv, at::Tensor& U,
                                                                        at::Tensor& S, at::Tensor& Vh);
at::Tensor& softplus_backward_out_common_nocheck(at::Tensor& grad_input, const at::Tensor& grad_output,
                                                 const at::Tensor& self, at::Scalar beta, at::Scalar threshold);
::std::tuple<at::Tensor, at::Tensor> triangular_solve_out_common_nocheck(const at::Tensor& self, const at::Tensor& A,
                                                                         bool upper, bool transpose, bool unitriangular);
at::Tensor& mean_out_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                    c10::optional<c10::ScalarType> dtype, at::Tensor& result);
at::Tensor mean_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                               c10::optional<c10::ScalarType> dtype);
at::Tensor& scatter_npu_common_nocheck(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src);
at::Tensor& scatter_npu_src_impl(at::Tensor& self, int64_t dim, const at::Tensor& index_ex, const at::Tensor& src_ex);
::std::tuple<at::Tensor, at::Tensor> prelu_backward_commom_nocheck(const at::Tensor& grad_output,
                                                                   const at::Tensor& self, const at::Tensor& weight);
at::Tensor prelu_common_nocheck(const at::Tensor& self, const at::Tensor& weight);
at::Tensor zeros_common_nocheck(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
                                c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                                c10::optional<bool> pin_memory_opt);
at::Tensor repeat_interleave_common_nocheck(const at::Tensor& self, int64_t repeats,
                                            c10::optional<int64_t> dim, c10::optional<int64_t> output_size);
at::Tensor repeat_interleave_common_nocheck(const at::Tensor& self, const at::Tensor& repeats,
                                            c10::optional<int64_t> dim, c10::optional<int64_t> output_size);
at::Tensor leaky_relu_backward_out_nocheck(at::Tensor result, const at::Tensor& grad_output, const at::Tensor& self,
                                           at::Scalar negval, bool is_result);
at::Tensor log_softmax_nocheck(at::Tensor& result, const at::Tensor& self, int64_t dim,
                               c10::optional<c10::ScalarType> dtype);
at::Tensor& cal_var_out(const at::Tensor& self, at::IntArrayRef dim, const int64_t correction, const bool unbiased,
                        const bool keepdim, at::Tensor& result);
at::Tensor cal_var(const at::Tensor& self, at::IntArrayRef dim, const int64_t correction, const bool unbiased,
                   const bool keepdim);
std::tuple<at::Tensor, at::Tensor> cal_var_mean(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
                                                int64_t correction, bool keepdim);
int64_t var_get_shape_prod(const at::Tensor& self, at::IntArrayRef dim);
std::tuple<at::Tensor, at::Tensor, at::Tensor> _svd_helper(const at::Tensor& self, bool some, bool compute_uv);
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
} // namespace acl_op
#endif
