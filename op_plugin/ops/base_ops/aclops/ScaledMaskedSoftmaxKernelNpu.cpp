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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using torch::autograd::AutogradContext;

namespace {

at::Tensor npu_scaled_masked_softmax_forward(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar scale,
    bool fixed_triu_mask) {
  at::Tensor result = npu_preparation::ApplyTensor(self, self.options().dtype(at::kHalf));
  at_npu::native::OpCommand cmd;
  cmd.Name("ScaledMaskedSoftmax")
      .Input(self)
      .Input(mask)
      .Output(result)
      .Attr("scale", scale)
      .Attr("fixed_triu_mask", fixed_triu_mask)
      .Run();
  return result;
}

at::Tensor npu_scaled_masked_softmax_backward(
    const at::Tensor& y_grad,
    const at::Tensor& y,
    const at::Tensor& mask,
    at::Scalar scale,
    bool fixed_triu_mask) {
  at::Tensor result = npu_preparation::ApplyTensor(y_grad, y_grad.options().dtype(at::kHalf));
  at_npu::native::OpCommand cmd;
  cmd.Name("ScaledMaskedSoftmaxGrad")
      .Input(y_grad)
      .Input(y)
      .Input(mask)
      .Output(result)
      .Attr("scale", scale)
      .Attr("fixed_triu_mask", fixed_triu_mask)
      .Run();
  return result;
}
} // namespace

class NPUscalemsFunction: public torch::autograd::Function<NPUscalemsFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self,
      const at::Tensor& mask,
      at::Scalar scale,
      bool fixed_triu_mask) {
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["fixed_triu_mask"] = fixed_triu_mask;
    auto result = npu_scaled_masked_softmax_forward(self, mask, scale, fixed_triu_mask);
    ctx->save_for_backward({result, mask});
    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto scale = ctx->saved_data["scale"].toScalar();
    auto fixed_triu_mask = ctx->saved_data["fixed_triu_mask"].toBool();
    auto saved = ctx->get_saved_variables();
    auto result0 = saved[0];
    auto mask = saved[1];
    auto result = npu_scaled_masked_softmax_backward(
        grad_outputs[0], result0, mask, scale, fixed_triu_mask);
    std::vector<at::Tensor> output = {result, at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor npu_scaled_masked_softmax(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& scale,
    bool fixed_triu_mask) {
  auto result = NPUscalemsFunction::apply(self, mask, scale, fixed_triu_mask);
  return result;
}

} // namespace acl_op
