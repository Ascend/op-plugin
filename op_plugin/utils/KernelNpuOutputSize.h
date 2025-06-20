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

#ifndef OP_PLUGIN_UTILS_KERNEL_NPU_INFER_SHAPE
#define OP_PLUGIN_UTILS_KERNEL_NPU_INFER_SHAPE

#include <ATen/ATen.h>

#include <string>
#include <tuple>
#include <vector>

#include "op_plugin/utils/Export.h"

namespace op_infer {
const int N = 32;
// npu tensor max size
const int SIZE = 8;
const int INT4_NUMS_IN_INT32_SPACE = 8;
const int NPU_NSA_COMPRESS_INPUT_DIM_SECOND = 1;
const int NPU_NSA_COMPRESS_INPUT_DIM_THIRD = 2;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

using tuple_array_vector = std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>;
using tuple_vector = std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;
using tuple_vectors =
    std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>;

inline c10::IntArrayRef infershape_for_elewise(const at::Tensor& x) { return x.sizes(); }

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> glu_npu_output_size(const at::Tensor& self, int64_t dim);

OP_PLUGIN_HIDDEN int64_t CeilDiv(int64_t value, int64_t factor);

OP_PLUGIN_HIDDEN int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);

OP_PLUGIN_HIDDEN c10::IntArrayRef input_same_output_size(const at::Tensor& input);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(c10::IntArrayRef shape1_,
                                                                               c10::IntArrayRef shape2_);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> broadcast_ops_npu_output_size(const at::Tensor& self,
                                                                               const at::Tensor& other);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> reduce_ops_npu_output_size(const at::Tensor& self,
                                                                            c10::IntArrayRef dim, bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> mse_loss_npu_output_size(const at::Tensor& self,
                                                                          const at::Tensor& target, int64_t reduction);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> adaptive_avg_pool3d_npu_output_size(const at::Tensor& self,
                                                                                     c10::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> addmm_npu_output_size(const at::Tensor& self, const at::Tensor& mat1,
                                                                       const at::Tensor& mat2);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> addbmm_npu_output_size(const at::Tensor& self,
                                                                        const at::Tensor& batch1,
                                                                        const at::Tensor& batch2);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> addmv_npu_output_size(const at::Tensor& self, const at::Tensor& mat);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> addr_npu_output_size(const at::Tensor& self, const at::Tensor& vec1,
                                                                      const at::Tensor& vec2);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> avg_pool2d_npu_output_size(
    const at::Tensor& self, c10::IntArrayRef kernel_size, c10::IntArrayRef stride, c10::IntArrayRef padding,
    bool ceil_mode);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> avg_pool3d_npu_output_size(
    const at::Tensor &self, c10::IntArrayRef kernel_size, c10::IntArrayRef stride, c10::IntArrayRef padding,
    bool ceil_mode);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> avg_pool2d_backward_npu_output_size(const at::Tensor& self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> baddbmm_npu_output_size(const at::Tensor& self,
                                                                         const at::Tensor& mat2);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> cdist_npu_output_size(const at::Tensor& x1, const at::Tensor& x2);

OP_PLUGIN_HIDDEN std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>
conv2d_backward_npu_output_size(const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight);

OP_PLUGIN_HIDDEN std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::SmallVector<int64_t, SIZE>>
conv2d_backward_tbc_output_size(const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> cosine_similarity_npu_output_size(const at::Tensor& x1, int64_t dim,
                                                                                   bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv1d_npu_output_size(const at::Tensor& input,
                                                                        const at::Tensor& weight,
                                                                        c10::IntArrayRef padding,
                                                                        c10::IntArrayRef stride,
                                                                        c10::IntArrayRef dilation);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv2d_npu_output_size(const at::Tensor& input,
                                                                        const at::Tensor& weight,
                                                                        c10::IntArrayRef padding,
                                                                        c10::IntArrayRef stride,
                                                                        c10::IntArrayRef dilation);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv_transpose1d_npu_output_size(
    const at::Tensor& input, const at::Tensor& weight, c10::IntArrayRef padding,
    c10::IntArrayRef output_padding, c10::IntArrayRef stride, c10::IntArrayRef dilation, int64_t groups);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
    const at::Tensor& input, const at::Tensor& weight, c10::IntArrayRef padding,
    c10::IntArrayRef output_padding, c10::IntArrayRef stride, c10::IntArrayRef dilation, int64_t groups);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv_npu_output_size(
    const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, c10::IntArrayRef padding,
    c10::IntArrayRef output_padding, c10::IntArrayRef stride, c10::IntArrayRef dilation, int64_t groups,
    bool transposed);

OP_PLUGIN_HIDDEN tuple_array_vector conv_transpose2d_backward_npu_output_size(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv_transpose2d_npu_output_size(
    const at::Tensor& input, const at::Tensor& weight, c10::IntArrayRef padding,
    c10::IntArrayRef output_padding, c10::IntArrayRef stride, c10::IntArrayRef dilation, int64_t groups);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> deformable_conv2d_npu_output_size(
    const at::Tensor& input, const at::Tensor& offset, c10::IntArrayRef kernel_size);

OP_PLUGIN_HIDDEN std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> ctc_loss_npu_output_size(
    const at::Tensor& log_probs, int64_t max_length);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> dot_npu_output_size();

OP_PLUGIN_HIDDEN std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nms_v4_npu_output_size(
    c10::Scalar max_output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> embedding_dense_backward_npu_output_size(
    const at::Tensor& grad_output, int64_t num_weights);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> index_npu_output_size(const at::Tensor& self, at::TensorList indices);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> index_select_npu_output_size(const at::Tensor& self, int64_t dim,
                                                                              const at::Tensor& index);

OP_PLUGIN_HIDDEN std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef> layer_norm_backward_npu_output_size(
    const at::Tensor& X, const at::Tensor& gamma);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> nnpack_spatial_convolution_npu_output_size(const at::Tensor& input,
                                                                                            const at::Tensor& weight,
                                                                                            c10::IntArrayRef padding,
                                                                                            c10::IntArrayRef stride);

OP_PLUGIN_HIDDEN tuple_vectors nms_with_mask_npu_output_size(const at::Tensor& self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> nonzero_npu_max_output_size(const at::Tensor& self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> prelu_backward_npu_grad_weight_output_size(const at::Tensor& weight);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> pad_npu_output_size(const at::Tensor& input,
                                                                     c10::IntArrayRef paddings);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> pdist_npu_output_size(const at::Tensor& self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor& self, int64_t dim,
                                                                      bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor& self, int64_t dim,
                                                                      bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> prod_npu_output_size(const at::Tensor& self, bool keepdim);

OP_PLUGIN_HIDDEN c10::IntArrayRef renorm_npu_output_size(const at::Tensor& self, c10::Scalar p, int dim,
                                                         c10::Scalar maxnorm);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(const at::Tensor& self,
                                                                                   int64_t repeats, int64_t dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size(const at::Tensor& self,
                                                                                   const at::Tensor& repeats,
                                                                                   int64_t dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> reflection_pad1d_npu_out_size(const at::Tensor& self,
                                                                               at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> reflection_pad2d_npu_out_size(const at::Tensor& self,
                                                                               at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> reflection_pad3d_npu_out_size(const at::Tensor& self,
                                                                               at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> replication_pad1d_npu_out_size(const at::Tensor& self,
                                                                                at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor& self,
                                                                                   c10::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_out_size(const at::Tensor& self,
                                                                                at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> conv_depthwise2d_npu_output_size(
    const at::Tensor& self, const at::Tensor& weight, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> replication_pad3d_npu_out_size(const at::Tensor& self,
                                                                                at::IntArrayRef padding);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> roi_align_backward_npu_output_size(c10::IntArrayRef xdiff_shape);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> repeat_npu_output_size(const at::Tensor& self,
                                                                        c10::IntArrayRef repeats);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> soft_margin_loss_npu_output_size(const at::Tensor& self,
                                                                                  int64_t reduction);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> slow_conv_dilated2d_npu_output_size(const at::Tensor& input,
                                                                                     const at::Tensor& weight,
                                                                                     c10::IntArrayRef stride,
                                                                                     c10::IntArrayRef padding,
                                                                                     c10::IntArrayRef dilation);

OP_PLUGIN_HIDDEN std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>
slow_conv_dilated2d_backward_npu_output_size(const at::Tensor& grad_output, const at::Tensor& self,
                                             const at::Tensor& weight);

OP_PLUGIN_HIDDEN std::tuple<c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef>
slow_conv_transpose2d_backward_npu_output_size(const at::Tensor& grad_output, const at::Tensor& self,
                                               const at::Tensor& weight);

OP_PLUGIN_HIDDEN c10::IntArrayRef smooth_l1_loss_npu_output_size(const at::Tensor& self, int64_t reduction);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> transpose_npu_output_size(const at::Tensor& self,
                                                                           c10::IntArrayRef perm);

OP_PLUGIN_HIDDEN tuple_vector softmax_cross_entropy_with_logits_impl_npu_output_size(const at::Tensor& self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> sum_npu_output_size(const at::Tensor& self, c10::IntArrayRef dim,
                                                                     bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> swiglu_backward_infershape(const at::Tensor &x, int64_t dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> topk_npu_output_size(const at::Tensor& self, int64_t k, int64_t dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, 3> upsample_infershape_with_scale(
    c10::IntArrayRef input_size, c10::optional<c10::IntArrayRef> output_size,
    c10::optional<c10::ArrayRef<double>> scale_factors);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> upsample_bicubic2d_npu_output_size(const at::Tensor& self,
                                                                                    c10::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::IntArrayRef upsample_bicubic2d_backward_npu_output_size(c10::IntArrayRef input_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> upsample_bilinear2d_npu_output_size(const at::Tensor& self,
                                                                                     c10::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> upsample_linear1d_npu_output_size(const at::Tensor& self,
                                                                                   c10::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> upsample_trilinear3d_npu_output_size(const at::Tensor& input,
                                                                                      at::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> upsample_nearest3d_npu_output_size(const at::Tensor& input,
                                                                                    at::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> var_npu_output_size(const at::Tensor& self, c10::IntArrayRef dim,
                                                                     bool keepdim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> crop_and_resize_npu_output_size(const at::Tensor& self,
                                                                                 at::IntArrayRef box_index,
                                                                                 at::IntArrayRef crop_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> decode_jpeg_npu_output_size(at::IntArrayRef image_shape,
                                                                             int64_t channels);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> infersize_stride_add(c10::IntArrayRef shape1_,
                                                                      c10::IntArrayRef shape2_);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> infersize_affine_grid_generator(at::IntArrayRef size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> infersize_all(const at::Tensor& self, int64_t dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> infersize_npu_anchor_response_flags(at::IntArrayRef featmap_size,
                                                                                     int64_t num_base_anchors);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> infersize_arange(const at::Scalar& start, const at::Scalar& end,
                                                                  const at::Scalar& step, at::ScalarType out_type);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> cat_npu_output_size(c10::SmallVector<at::Tensor, N>& tensors,
                                                                     int64_t dimension);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> clamp_npu_output_size(const at::Tensor& self,
                                                                       const c10::optional<at::Tensor>& min,
                                                                       const c10::optional<at::Tensor>& max);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> image_to_col_npu_output_size(const at::Tensor& self,
                                                                              at::IntArrayRef ksizes,
                                                                              at::IntArrayRef strides,
                                                                              at::IntArrayRef dilations,
                                                                              at::IntArrayRef pads);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> max_pool2d_out_size(const at::Tensor& self,
                                                                     at::IntArrayRef output_size);
OP_PLUGIN_HIDDEN std::vector<c10::SmallVector<int64_t, SIZE>> rms_norm_npu_output_size(const at::Tensor &self,
                                                                                       const at::Tensor &gamma);

OP_PLUGIN_HIDDEN std::vector<c10::SmallVector<int64_t, SIZE>> rms_norm_grad_npu_output_size(const at::Tensor &self,
                                                                                            const at::Tensor &gamma);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> ger_output_size(const at::Tensor& self, const at::Tensor& vec2);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> im2col_backward_npu_output_size(const at::Tensor& grad_output,
                                                                                 const at::IntArrayRef& input_size,
                                                                                 const at::IntArrayRef& kernel_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size_opapi(const at::Tensor& self,
                                                                                         int64_t repeats,
                                                                                         c10::optional<int64_t> dim);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> repeat_interleave_npu_output_size_opapi(const at::Tensor& self,
                                                                                         const at::Tensor& repeats,
                                                                                         c10::optional<int64_t> dim,
                                                                                         c10::optional<int64_t> output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> max_pool3d_output_size(const at::Tensor& self,
                                                                        at::IntArrayRef output_size);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> diag_output_size(const at::Tensor& self,
                                                                  int64_t diagonal);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> stack_output_size(at::TensorList tensors,
                                                                   int64_t dim);

OP_PLUGIN_HIDDEN at::SmallVector<int64_t, SIZE> upsample_nearest_exact2d_output_size_npu(const at::Tensor &input,
                                                                                         at::IntArrayRef output_size);

OP_PLUGIN_HIDDEN at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_loss_output_size(const at::Tensor &input,
                                                                                        c10::string_view reduction);

OP_PLUGIN_HIDDEN at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_zloss_output_size(const at::Tensor &input,
                                                                                         c10::string_view reduction,
                                                                                         bool return_zloss);

OP_PLUGIN_HIDDEN at::SmallVector<int64_t, SIZE> npu_cross_entropy_loss_lse_for_zloss_output_size(const at::Tensor &input,
                                                                                                 float lse_square_scale_for_zloss);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> kronecker_quant_out_size(const at::Tensor &self);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> kronecker_quant_scale_size(const at::Tensor &self);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> matmul_output_size(const at::Tensor &tensor1, const at::Tensor &tensor2);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_transpose_batchmatmul_output_size(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &scale_real,
                                                                                       at::IntArrayRef perm_x1_real, at::IntArrayRef perm_x2_real, at::IntArrayRef perm_y_real,
                                                                                       int32_t batch_split_factor_value);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_group_quant_out_size(const at::Tensor& x, c10::optional<at::ScalarType> dst_dtype);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_gather_sparse_index_out_size(const at::Tensor& input, const at::Tensor& index);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_nsa_compress_out_size(const at::Tensor& input, c10::optional<int64_t> actual_seq_len_type, at::OptionalIntArrayRef actual_seq_len, int64_t compress_block_size, int64_t compress_stride);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_nsa_select_attention_infer_out_size(const at::Tensor& query, const at::Tensor& value, int64_t head_num, int64_t key_value_head_num, c10::string_view layout);

OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_moe_token_permute_out_size(const at::Tensor &tokens, const at::Tensor &indices, c10::optional<int64_t> num_out_tokens);
OP_PLUGIN_HIDDEN c10::SmallVector<int64_t, SIZE> npu_moe_token_unpermute_out_size(const at::Tensor& permuted_tokens, const at::Tensor &sorted_indices, const c10::optional<at::Tensor>& probs);

} // namespace op_infer
#endif // OP_PLUGIN_UTILS_KERNEL_NPU_INFER_SHAPE
