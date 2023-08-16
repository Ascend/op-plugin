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
using torch::autograd::Function;

namespace{
at::Tensor confusion_transpose_npu(
    const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  c10::SmallVector<int64_t, SIZE> output_size;
  if (transpose_first) {
    output_size = op_infer::array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++) {
      output_size.emplace_back(shape[perm[i]]);
    }
  }

  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  at_npu::native::OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(self)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", shape)
      .Attr("transpose_first", transpose_first)
      .Run();

  return result;
}
} // namespace

at::Tensor npu_confusion_transpose_backward(
    const at::Tensor& grad,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  c10::SmallVector<int64_t, SIZE> svec_shape;
  if (transpose_first) {
    svec_shape = op_infer::array_to_small_vector(shape);
  } else {
    for (int i = 0; i < perm.size(); i++) {
      svec_shape.emplace_back(shape[perm[i]]);
    }
  }
  std::vector<int64_t> vec_perm;
  int64_t perm_len = perm.size();
  int64_t temp_perm[perm_len] = {0};
  for (int64_t i = 0; i < perm_len; i++) {
    temp_perm[perm[i]] = i;
  }
  vec_perm = std::vector<int64_t>(temp_perm, temp_perm+perm_len);
  perm = at::IntArrayRef(vec_perm);
  at::Tensor result = npu_preparation::ApplyTensor(grad, shape);

  at_npu::native::OpCommand cmd;
  cmd.Name("ConfusionTransposeD")
      .Input(grad)
      .Output(result)
      .Attr("perm", perm)
      .Attr("shape", svec_shape)
      .Attr("transpose_first", transpose_first)
      .Run();
  return result;
}

class NPUConfusionTransposeFunction : public torch::autograd::Function<NPUConfusionTransposeFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self,
      at::IntArrayRef perm,
      at::IntArrayRef shape,
      bool transpose_first) {
    ctx->saved_data["perm"] = perm;
    ctx->saved_data["shape"] = self.sizes();
    ctx->saved_data["transpose_first"] = !transpose_first;
    at::AutoNonVariableTypeMode g;
    return confusion_transpose_npu(self, perm, shape, transpose_first);
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto perm = ctx->saved_data["perm"].toIntVector();
    auto shape = ctx->saved_data["shape"].toIntVector();
    auto transpose_first = ctx->saved_data["transpose_first"].toBool();
    at::Tensor result = acl_op::npu_confusion_transpose_backward(grad_outputs[0], perm, shape, transpose_first);

    std::vector<at::Tensor> output = {result, at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor npu_confusion_transpose(const at::Tensor& self,
    at::IntArrayRef perm,
    at::IntArrayRef shape,
    bool transpose_first) {
  return NPUConfusionTransposeFunction::apply(self, perm, shape, transpose_first);
}
} // namespace acl_op
