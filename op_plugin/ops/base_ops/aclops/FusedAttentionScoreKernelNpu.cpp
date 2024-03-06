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

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;

namespace {
std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> fused_attention_score_infer_shape(
    const at::Tensor& query_layer,
    const at::Tensor& attention_mask) {
  c10::SmallVector<int64_t, SIZE> attention_score_output_shape = {
      query_layer.size(0) * query_layer.size(2), query_layer.size(1) * query_layer.size(3)};
  c10::SmallVector<int64_t, SIZE> softmax_output_shape = {
      query_layer.size(0), query_layer.size(1), query_layer.size(2), query_layer.size(2)};
  return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
      attention_score_output_shape, softmax_output_shape);
}

at::Tensor dropout_gen_mask_nocheck(const at::Tensor& self, const at::Scalar& prob) {
  at::Tensor mask = npu_preparation::apply_tensor_with_format(
      {self.numel()},
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  const int64_t seed = static_cast<int64_t>(gen.current_seed());
  const int64_t seed2 = 0;
  at_npu::native::OpCommand cmd;
  cmd.Name("DropOutGenMaskV3")
      .Input(self.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
  return mask;
}

std::tuple<at::Tensor&, at::Tensor&> npu_fused_attention_score_nocheck(
    at::Tensor& attention_score,
    at::Tensor& softmax_output,
    const at::Tensor& query_layer,
    const at::Tensor& key_layer,
    const at::Tensor& value_layer,
    const at::Tensor& attention_mask,
    const at::Tensor& drop_mask,
    const at::Scalar& scale,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool bmm_score_transpose_a,
    bool bmm_score_transpose_b) {
  at_npu::native::OpCommand cmd;
  cmd.Name("AttentionScore")
      .Input(query_layer)
      .Input(key_layer)
      .Input(value_layer)
      .Input(attention_mask)
      .Input(scale, at::kHalf)
      .Input(drop_mask)
      .Output(attention_score)
      .Output(softmax_output)
      .Attr("keep_prob", (float)keep_prob)
      .Attr("query_transpose", query_transpose)
      .Attr("key_transpose", key_transpose)
      .Attr("bmm_score_transpose_a", bmm_score_transpose_a)
      .Attr("bmm_score_transpose_b", bmm_score_transpose_b)
      .Run();
  return std::tie(attention_score, softmax_output);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_backward(
    const at::Tensor& grad_output,
    const at::Tensor& softmax_output,
    const at::Tensor& query_layer,
    const at::Tensor& key_layer,
    const at::Tensor& value_layer,
    const at::Tensor& drop_mask,
    const at::Scalar& scale,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool value_transpose,
    bool dx_transpose) {
  at::Tensor query_dx = npu_preparation::apply_tensor(grad_output);
  at::Tensor key_dw = npu_preparation::apply_tensor(grad_output);
  at::Tensor value_dw = npu_preparation::apply_tensor(grad_output);
  at::Tensor grad_output_permute = grad_output.reshape(
      {query_layer.size(0), query_layer.size(2), query_layer.size(1), query_layer.size(3)}).permute({0, 2, 1, 3});

  at_npu::native::OpCommand cmd;
  cmd.Name("AttentionScoreGrad")
      .Input(softmax_output)
      .Input(grad_output_permute)
      .Input(value_layer)
      .Input(key_layer)
      .Input(query_layer)
      .Input(scale, at::kHalf)
      .Input(drop_mask)
      .Output(value_dw)
      .Output(query_dx)
      .Output(key_dw)
      .Attr("keep_prob", (float)keep_prob)
      .Attr("query_transpose", query_transpose)
      .Attr("key_transpose", key_transpose)
      .Attr("value_transpose", value_transpose)
      .Attr("dx_transpose", dx_transpose)
      .Run();
  query_dx = query_dx.reshape({query_layer.size(0), query_layer.size(2), query_layer.size(1), query_layer.size(3)})
                     .permute({0, 2, 1, 3});
  key_dw = key_dw.reshape({query_layer.size(0), query_layer.size(2), query_layer.size(1), query_layer.size(3)})
                     .permute({0, 2, 1, 3});
  value_dw = value_dw.reshape({query_layer.size(0), query_layer.size(2), query_layer.size(1), query_layer.size(3)})
                     .permute({0, 2, 1, 3});
  return std::tie(query_dx, key_dw, value_dw);
}

at::Tensor npu_fused_attention_score(
    const at::Tensor& query_layer,
    const at::Tensor& key_layer,
    const at::Tensor& value_layer,
    const at::Tensor& attention_mask,
    const at::Scalar& scale,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool bmm_score_transpose_a,
    bool bmm_score_transpose_b,
    bool value_transpose,
    bool dx_transpose) {
    TORCH_CHECK(query_layer.dim() >= 4, "query_layer must be at least 4-dimensional"
        + OPS_ERROR(ErrCode::PARAM));
  auto results = at_npu::native::custom_ops::npu_fused_attention_score_fwd(
      query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose,
      key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
  return std::get<0>(results);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_fwd(
    const at::Tensor& query_layer,
    const at::Tensor& key_layer,
    const at::Tensor& value_layer,
    const at::Tensor& attention_mask,
    const at::Scalar& scale,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool bmm_score_transpose_a,
    bool bmm_score_transpose_b,
    bool value_transpose,
    bool dx_transpose) {
  auto output_sizes = fused_attention_score_infer_shape(query_layer, attention_mask);
  at::Tensor attention_score = npu_preparation::apply_tensor(query_layer, std::get<0>(output_sizes));
  at::Tensor softmax_output = npu_preparation::apply_tensor(query_layer, std::get<1>(output_sizes));
  at::Tensor drop_mask;
  auto original_stream = c10_npu::getCurrentNPUStream();
  {
    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
    drop_mask = dropout_gen_mask_nocheck(softmax_output, at::Scalar(keep_prob));
  }
  c10_npu::NPUCachingAllocator::recordStream(drop_mask.storage().data_ptr(), original_stream);
  npu_fused_attention_score_nocheck(attention_score, softmax_output, query_layer, key_layer, value_layer,
      attention_mask, drop_mask, scale, keep_prob, query_transpose, key_transpose,
      bmm_score_transpose_a, bmm_score_transpose_b);
  return std::tie(attention_score, softmax_output, drop_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_grad(
    const at::Tensor& grad_output,
    const at::Tensor& softmax_output,
    const at::Tensor& query_layer,
    const at::Tensor& key_layer,
    const at::Tensor& value_layer,
    const at::Tensor& drop_mask,
    const at::Scalar& scale,
    double keep_prob,
    bool query_transpose,
    bool key_transpose,
    bool value_transpose,
    bool dx_transpose) {
    TORCH_CHECK(query_layer.dim() >= 4, "query_layer must be at least 4-dimensional"
        + OPS_ERROR(ErrCode::PARAM));
    at::Tensor query_dx = npu_preparation::apply_tensor_with_format(grad_output, ACL_FORMAT_FRACTAL_NZ);
    at::Tensor key_dw = npu_preparation::apply_tensor_with_format(grad_output, ACL_FORMAT_FRACTAL_NZ);
    at::Tensor value_dw = npu_preparation::apply_tensor_with_format(grad_output, ACL_FORMAT_FRACTAL_NZ);
    at::Tensor grad_output_permute = acl_op::npu_confusion_transpose(grad_output, {0, 2, 1, 3},
        {query_layer.size(0), query_layer.size(2), query_layer.size(1), query_layer.size(3)}, false);
    at_npu::native::OpCommand cmd;
    cmd.Name("AttentionScoreGrad")
        .Input(softmax_output)
        .Input(grad_output_permute)
        .Input(value_layer)
        .Input(key_layer)
        .Input(query_layer)
        .Input(scale, at::kHalf)
        .Input(drop_mask)
        .Output(value_dw)
        .Output(query_dx)
        .Output(key_dw)
        .Attr("keep_prob", (float)keep_prob)
        .Attr("query_transpose", query_transpose)
        .Attr("key_transpose", key_transpose)
        .Attr("value_transpose", value_transpose)
        .Attr("dx_transpose", dx_transpose)
        .Run();
    return std::tie(query_dx, key_dw, value_dw);
}
} // namespace acl_op
