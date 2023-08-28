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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& _log_softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& result) {
  c10::ScalarType result_type = half_to_float ? c10::ScalarType::Float : result.scalar_type();
  npu_preparation::CheckOut(
    {self},
    result,
    calcu_op_util::GetTensorNpuFormat(result),
    result_type,
    self.sizes());

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    log_softmax_nocheck(contiguous_result, self, dim, result_type);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    log_softmax_nocheck(result, self, dim, result_type);
  }
  return result;
}
} // namespace acl_op
