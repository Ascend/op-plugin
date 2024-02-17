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
at::Tensor& celu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Scalar& alpha) {
  at_npu::native::OpCommand cmd;
  cmd.Name("CeluV2")
      .Input(self)
      .Output(result)
      .Attr("alpha", alpha)
      .Run();
  return result;
}

at::Tensor celu_out_nocheck(const at::Tensor& self, const at::Scalar& alpha) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  celu_out_npu_nocheck(result, self, alpha);
  return result;
}
} // namespace

at::Tensor celu(const at::Tensor& self, const at::Scalar& alpha) {
  return celu_out_nocheck(self, alpha);
}

at::Tensor& celu_(at::Tensor& self, const at::Scalar& alpha) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    at::Tensor result = celu_out_nocheck(contiguous_self, alpha);
    npu_utils::format_fresh_view(self, result);
  } else {
    auto result = celu_out_nocheck(self, alpha);
    self.copy_(result);
  }
  return self;
}
} // namespace acl_op
