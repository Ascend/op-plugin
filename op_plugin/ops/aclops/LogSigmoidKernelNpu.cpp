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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&> log_sigmoid_forward_out_nocheck(
    at::Tensor& output,
    at::Tensor& buffer,
    const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LogSigmoid")
      .Input(self)
      .Output(output)
      .Run();
  return std::tie(output, buffer);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> log_sigmoid_forward_out(
    const at::Tensor& self,
    at::Tensor& output,
    at::Tensor& buffer) {
  npu_preparation::CheckOut({self}, output, self);
  npu_preparation::CheckOut({self}, buffer, self);
  if (!npu_utils::check_match(&output)) {
    at::Tensor contig_tensor = npu_utils::format_contiguous(output);
    log_sigmoid_forward_out_nocheck(contig_tensor, buffer, self);
    npu_utils::format_fresh_view(output, contig_tensor);
  } else {
    log_sigmoid_forward_out_nocheck(output, buffer, self);
  }
  return std::tie(output, buffer);
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(const at::Tensor& self) {
  at::Tensor output = npu_preparation::apply_tensor(self);
  at::Tensor buffer = npu_preparation::ApplyTensorWithSizes({0}, self.options());
  log_sigmoid_forward_out_nocheck(output, buffer, self);
  return std::tie(output, buffer);
}

at::Tensor& log_sigmoid_out(const at::Tensor& self, at::Tensor& result) {
  at::Tensor buffer = npu_preparation::ApplyTensorWithSizes({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(result, buffer, self));
}

at::Tensor log_sigmoid(const at::Tensor& self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

} // namespace acl_op
