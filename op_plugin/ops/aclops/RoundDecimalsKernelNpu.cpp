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

static void round_decimals_check(const at::Tensor& self, int64_t decimals) {
  if (isIntegralType(self.scalar_type(), true)) {
    TORCH_CHECK(decimals == 0, "round_npu not implemented for ", toString(self.scalar_type()), " with decimals != 0"
        + OPS_ERROR(ErrCode::VALUE));
  }
}

at::Tensor& round_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, int64_t decimals) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Round")
      .Input(self)
      .Output(result)
      .Attr("decimals", decimals)
      .Run();

  return result;
}
} // namespace

at::Tensor& round_out(const at::Tensor& self, int64_t decimals, at::Tensor& result) {
  round_decimals_check(self, decimals);
  npu_preparation::CheckOut({self}, result, self);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    round_out_npu_nocheck(contiguous_result, self, decimals);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    round_out_npu_nocheck(result, self, decimals);
  }

  return result;
}

at::Tensor round(const at::Tensor& self, int64_t decimals) {
  round_decimals_check(self, decimals);
  at::Tensor result = npu_preparation::ApplyTensor(self);
  round_out_npu_nocheck(result, self, decimals);

  return result;
}

at::Tensor& round_(at::Tensor& self, int64_t decimals) {
  round_decimals_check(self, decimals);
  acl_op::round_out(self, decimals, self);

  return self;
}

} // namespace acl_op
