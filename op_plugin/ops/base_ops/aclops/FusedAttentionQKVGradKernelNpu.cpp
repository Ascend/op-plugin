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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> npu_fused_attention_qkv_grad(
    const at::Tensor& grad_output_query,
    const at::Tensor& grad_output_key,
    const at::Tensor& grad_output_value,
    const at::Tensor& query_kernel,
    const at::Tensor& key_kernel,
    const at::Tensor& value_kernel,
    const at::Tensor& hidden_states,
    const at::Tensor& grad_output_ln) {
  at::Tensor grad_hidden_states = npu_preparation::ApplyTensorWithFormat(hidden_states, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor grad_w_query = npu_preparation::ApplyTensor(query_kernel);
  at::Tensor grad_w_key = npu_preparation::ApplyTensor(key_kernel);
  at::Tensor grad_w_value = npu_preparation::ApplyTensor(value_kernel);
  at::Tensor grad_b_query = npu_preparation::ApplyTensorWithFormat(query_kernel, query_kernel.size(0), ACL_FORMAT_ND);
  at::Tensor grad_b_key = npu_preparation::ApplyTensorWithFormat(key_kernel, key_kernel.size(0), ACL_FORMAT_ND);
  at::Tensor grad_b_value = npu_preparation::ApplyTensorWithFormat(value_kernel, value_kernel.size(0), ACL_FORMAT_ND);

  at_npu::native::OpCommand cmd;
  cmd.Name("AttentionQKVGradX")
      .Input(grad_output_ln)
      .Input(grad_output_query)
      .Input(grad_output_key)
      .Input(grad_output_value)
      .Input(query_kernel)
      .Input(key_kernel)
      .Input(value_kernel)
      .Output(grad_hidden_states)
      .Attr("trans_a", false)
      .Attr("trans_b", true)
      .Run();
  cmd.Name("AttentionQKVGradW")
      .Input(hidden_states)
      .Input(grad_output_query)
      .Input(grad_output_key)
      .Input(grad_output_value)
      .Output(grad_w_query)
      .Output(grad_w_key)
      .Output(grad_w_value)
      .Output(grad_b_query)
      .Output(grad_b_key)
      .Output(grad_b_value)
      .Attr("trans_a", true)
      .Attr("trans_b", false)
      .Run();

  std::vector<at::Tensor> results = {
      grad_hidden_states, grad_w_query, grad_w_key, grad_w_value, grad_b_query, grad_b_key, grad_b_value};
  return results;
}
} // namespace acl_op
