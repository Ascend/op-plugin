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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace op_api {

// norm.out
at::Tensor& norm_out(
    const at::Tensor &self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor &out) {
  DO_COMPATIBILITY(aclnnNorm, acl_op::norm_out(self, p, dim, keepdim, out));
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  at_npu::native::OpPreparation::check_tensor({self}, out, self.scalar_type(), outputSize);
  
  at::Scalar pvalue = 2;
  if (p.has_value()) {
    pvalue = p.value();
  }
  EXEC_NPU_CMD(aclnnNorm, self, pvalue, dim, keepdim, out);
  return out;
}

} // namespace op_api