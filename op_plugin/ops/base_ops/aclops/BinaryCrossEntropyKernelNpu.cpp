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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& binary_cross_entropy_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction) {
  at::Tensor weight_tensor = weight.defined() ? weight : at::ones(self.sizes(), self.options());
  std::string reduction_str = op_plugin::utils::get_reduction_str(reduction);

  at_npu::native::OpCommand cmd;
  cmd.Name("BinaryCrossEntropy")
      .Input(self)
      .Input(target)
      .Input(weight_tensor)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Run();
  return result;
}
} // namespace

at::Tensor& binary_cross_entropy_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    at::Tensor& result) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = self.sizes();
  } else {
    output_size = at::ArrayRef<int64_t>();
  }
  if (self.numel() == 0) {
    at::Tensor result_cp = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat).fill_(NAN);
    result.copy_(result_cp);
    return result;
  }
  npu_preparation::CheckOut(
      {self, target, weight},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    binary_cross_entropy_out_nocheck(contiguous_result, self, target, weight, reduction);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    binary_cross_entropy_out_nocheck(result, self, target, weight, reduction);
  }
  return result;
}

at::Tensor binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = self.sizes();
  } else {
    output_size = at::ArrayRef<int64_t>();
  }

  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  if (self.numel() == 0) {
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat).fill_(NAN);
    return result;
  }

  binary_cross_entropy_out_nocheck(result, self, target, weight, reduction);
  return result;
}
} // namespace acl_op
