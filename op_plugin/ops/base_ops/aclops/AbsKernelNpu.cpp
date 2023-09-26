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
at::Tensor& abs_out_nocheck(at::Tensor& result, const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Abs")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}
}

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& result) {
  npu_preparation::CheckOut({self}, result, self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    abs_out_nocheck(contiguous_result, self);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    abs_out_nocheck(result, self);
  }
  return result;
}

at::Tensor abs(const at::Tensor& self) {
  auto output_size = op_infer::infershape_for_elewise(self);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  abs_out_nocheck(result, self);
  return result;
}

at::Tensor& abs_(at::Tensor& self) {
  return acl_op::abs_out(self, self);
}
} // namespace acl_op
