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

#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

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
  at::Tensor mask = npu_preparation::ApplyTensorWithFormat(
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
  at::Tensor query_dx = npu_preparation::ApplyTensor(grad_output);
  at::Tensor key_dw = npu_preparation::ApplyTensor(grad_output);
  at::Tensor value_dw = npu_preparation::ApplyTensor(grad_output);
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

class NPUFusedAttentionScoreFunction : public torch::autograd::Function<NPUFusedAttentionScoreFunction> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx,
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
    at::AutoNonVariableTypeMode g;
    auto output_sizes = fused_attention_score_infer_shape(query_layer, attention_mask);
    at::Tensor attention_score = npu_preparation::ApplyTensor(query_layer, std::get<0>(output_sizes));
    at::Tensor softmax_output = npu_preparation::ApplyTensor(query_layer, std::get<1>(output_sizes));
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
    std::vector<at::Tensor> result_list = {attention_score, softmax_output};

    ctx->save_for_backward({softmax_output, query_layer, key_layer, value_layer, drop_mask});
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["keep_prob"] = keep_prob;
    ctx->saved_data["query_transpose"] = query_transpose;
    ctx->saved_data["key_transpose"] = key_transpose;
    ctx->saved_data["value_transpose"] = value_transpose;
    ctx->saved_data["dx_transpose"] = dx_transpose;
    return result_list;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto scale = ctx->saved_data["scale"].toScalar();
    auto keep_prob = ctx->saved_data["keep_prob"].toDouble();
    auto query_transpose = ctx->saved_data["query_transpose"].toBool();
    auto key_transpose = ctx->saved_data["key_transpose"].toBool();
    auto value_transpose = ctx->saved_data["value_transpose"].toBool();
    auto dx_transpose = ctx->saved_data["dx_transpose"].toBool();
    auto saved = ctx->get_saved_variables();
    auto softmax_output = saved[0];
    auto query_layer = saved[1];
    auto key_layer = saved[2];
    auto value_layer = saved[3];
    auto drop_mask = saved[4];
    auto result = acl_op::npu_fused_attention_score_backward(
        grad_outputs[0], softmax_output, query_layer, key_layer, value_layer, drop_mask, scale,
        keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
    std::vector<at::Tensor> output = {
        std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

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
  auto result = NPUFusedAttentionScoreFunction::apply(
      query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose,
      key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
  return result[0];
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
  at::Tensor attention_score = npu_preparation::ApplyTensor(query_layer, std::get<0>(output_sizes));
  at::Tensor softmax_output = npu_preparation::ApplyTensor(query_layer, std::get<1>(output_sizes));
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
  at::Tensor query_dx = npu_preparation::ApplyTensorWithFormat(grad_output, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor key_dw = npu_preparation::ApplyTensorWithFormat(grad_output, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor value_dw = npu_preparation::ApplyTensorWithFormat(grad_output, ACL_FORMAT_FRACTAL_NZ);
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
