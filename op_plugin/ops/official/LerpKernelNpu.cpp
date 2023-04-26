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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& lerp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Lerp")
      .Input(self)
      .Input(end)
      .Input(weight)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& lerp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& end,
    at::Scalar weight) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Lerp")
      .Input(self)
      .Input(end)
      .Input(weight, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& lerp_out(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self, end, weight},
      result,
      self);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    lerp_out_npu_nocheck(contiguous_result, self, end, weight);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    lerp_out_npu_nocheck(result, self, end, weight);
  }
  return result;
}

at::Tensor& lerp_out(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self, end},
      result,
      self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    lerp_out_npu_nocheck(contiguous_result, self, end, weight);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    lerp_out_npu_nocheck(result, self, end, weight);
  }
  return result;
}

at::Tensor lerp(const at::Tensor& start, const at::Tensor& end, const at::Tensor& weight) {
  at::Tensor result = npu_preparation::ApplyTensor(start);
  lerp_out_npu_nocheck(result, start, end, weight);
  return result;
}

at::Tensor lerp(const at::Tensor& start, const at::Tensor& end, const at::Scalar& weight) {
  at::Tensor result = npu_preparation::ApplyTensor(start);
  lerp_out_npu_nocheck(result, start, end, weight);
  return result;
}

at::Tensor& lerp_(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  return op_plugin::lerp_out(self, end, weight, self);
}

at::Tensor& lerp_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
  return op_plugin::lerp_out(self, end, weight, self);
}
} // namespace op_plugin
