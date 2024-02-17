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
at::Tensor& threshold_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Scalar& value) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ThresholdV2D")
      .Input(self)
      .Output(result)
      .Attr("threshold", threshold)
      .Attr("value", value)
      .Run();
  return result;
}
} // namespace

at::Tensor& threshold_out(
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Scalar& value,
    at::Tensor& result) {
  npu_preparation::CheckOut({self}, result, self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(result);
    threshold_out_nocheck(contiguous_self, self, threshold, value);
    npu_utils::format_fresh_view(result, contiguous_self);
  } else {
    threshold_out_nocheck(result, self, threshold, value);
  }

  return result;
}

at::Tensor threshold(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  threshold_out_nocheck(result, self, threshold, value);
  return result;
}

at::Tensor& threshold_(at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  return acl_op::threshold_out(self, threshold, value, self);
}
} // namespace acl_op
