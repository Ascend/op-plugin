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
at::Tensor& leaky_relu_out_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar negval) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LeakyRelu")
      .Input(self)
      .Output(result)
      .Attr("negative_slope", negval)
      .Run();

  return result;
}
} // namespace

at::Tensor& leaky_relu_out(const at::Tensor& self, const at::Scalar& negval, at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);
  
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    leaky_relu_out_nocheck(contiguous_result, self, negval);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    leaky_relu_out_nocheck(result, self, negval);
  }
  return result;
}

at::Tensor leaky_relu(const at::Tensor& self, const at::Scalar& negval) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  leaky_relu_out_nocheck(result, self, negval);
  return result;
}

at::Tensor& leaky_relu_(at::Tensor& self, const at::Scalar& neg_val) {
  return op_plugin::leaky_relu_out(self, neg_val, self);
}
} // namespace op_plugin
