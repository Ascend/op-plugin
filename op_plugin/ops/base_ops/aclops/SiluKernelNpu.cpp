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
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& silu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Swish")
      .Input(self)
      .Output(result)
      .Attr("scale", (float)1.0)
      .Run();
  return result;
}

at::Tensor& silu_out_npu(const at::Tensor& self, at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);
    
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    silu_out_npu_nocheck(contiguous_result, self);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    silu_out_npu_nocheck(result, self);
  }

  return result;
}

at::Tensor silu_kernel_npu(const at::Tensor& self) {
  at::Tensor result = npu_preparation::ApplyTensor(self);

  silu_out_npu_nocheck(result, self);

  return result;
}

at::Tensor& silu_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& x0,
    const at::Tensor& x1) {

  at_npu::native::OpCommand cmd;
  cmd.Name("SwishGrad")
      .Input(grad_output)
      .Input(x0)
      .Input(x1)
      .Output(result)
      .Run();

  return result;
}

} // namespace

at::Tensor& npu_silu_(at::Tensor& self) {
  silu_out_npu(self, self);
  return self;
}

at::Tensor npu_silu_backward(const at::Tensor& grad_output, const at::Tensor& x0, const at::Tensor& x1) {
  at::Tensor grad_input = npu_preparation::ApplyTensor(grad_output);
  silu_backward_out_npu_nocheck(grad_input, grad_output, x0, x1);

  return grad_input;
}

class NPUSiluFunction : public torch::autograd::Function<NPUSiluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    at::Tensor result = silu_kernel_npu(self);
    ctx->save_for_backward({self, result});

    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto result = saved[1];

    at::Tensor output = acl_op::npu_silu_backward(grad_outputs[0], input, result);
    std::vector<at::Tensor> output_list = {output};

    return output_list;
  }
};

at::Tensor npu_silu(const at::Tensor& self) {
  return NPUSiluFunction::apply(self);
}

at::Tensor& silu_out(const at::Tensor& self, at::Tensor& result) {
  silu_out_npu(self, result);
  return result;
}

at::Tensor silu(const at::Tensor& self) {
  return NPUSiluFunction::apply(self);
}

class NPUSiluInplaceFunction : public torch::autograd::Function<NPUSiluInplaceFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    at::Tensor self_copy = self.clone();
    at::Tensor result = silu_kernel_npu(self);
    ctx->save_for_backward({self_copy, result});

    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto result = saved[1];

    at::Tensor output = acl_op::npu_silu_backward(grad_outputs[0], input, result);
    std::vector<at::Tensor> output_list = {output};

    return output_list;
  }
};

at::Tensor& silu_(at::Tensor& self) {
  at::Tensor result = NPUSiluInplaceFunction::apply(self);
  self.copy_(result);
  return self;
}

} // namespace acl_op
