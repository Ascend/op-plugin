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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& histc_out(const at::Tensor& self, int64_t bins, const at::Scalar& min, 
                      const at::Scalar& max, at::Tensor& result) {
  at_npu::native::OpPreparation::check_tensor({self}, result, self.scalar_type(), {bins});
  EXEC_NPU_CMD(aclnnHistc, self, bins, min, max, result);
  return result;
}

at::Tensor histc(const at::Tensor& self, int64_t bins, const at::Scalar& min, 
                 const at::Scalar& max) {
  at::ScalarType out_type = self.scalar_type();
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format({bins}, 
                      self.options().dtype(out_type));
  EXEC_NPU_CMD(aclnnHistc, self, bins, min, max, result);
  return result;
}

}  // namespace op_api
