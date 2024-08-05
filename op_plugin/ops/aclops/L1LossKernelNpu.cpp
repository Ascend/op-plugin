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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& l1_loss_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    const int64_t reduction) {
    std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("LpLoss")
        .Input(self)
        .Input(target)
        .Attr("reduction", reduction_str)
        .Attr("p", (int64_t)1)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& l1_loss_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const int64_t reduction) {
    std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);
    at_npu::native::OpCommand cmd;
    cmd.Name("L1LossGrad")
        .Input(grad_output)
        .Input(self)
        .Input(target)
        .Attr("reduction", reduction_str)
        .Output(grad_input)
        .Run();
    return grad_input;
}
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& l1_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self, target},
      result,
      self);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    l1_loss_out_nocheck(contiguous_result, self, target, reduction);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    l1_loss_out_nocheck(result, self, target, reduction);
  }
  return result;
}

at::Tensor l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::input_same_output_size(self);
  }
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  l1_loss_out_nocheck(result, self, target, reduction);
  return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
    at::Tensor grad_output_broadcast = grad_output;
    at::Tensor target_broadcast = target;
    if (grad_output.sizes() != self.sizes()) {
        grad_output_broadcast = acl_op::npu_broadcast(grad_output, self.sizes());
    }
    if (target.sizes() != self.sizes()) {
        target_broadcast = acl_op::npu_broadcast(target, self.sizes());
    }
    at::Tensor result = npu_preparation::apply_tensor(self);
    l1_loss_backward_out_nocheck(result, grad_output_broadcast, self, target_broadcast, reduction);
    return result;
}

at::Tensor npu_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
    at::IntArrayRef output_size;
    if (reduction == at::Reduction::None) {
        output_size = op_infer::input_same_output_size(self);
    }
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    l1_loss_out_nocheck(result, self, target, reduction);
    return result;
}

at::Tensor l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
    return npu_l1_loss(self, target, reduction);
}
#endif
} // namespace acl_op
