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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& max_unpool2d_out(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef outputSize,
    at::Tensor& output) {
  DO_COMPATIBILITY(aclnnMaxUnpool2d, acl_op::max_unpool2d_out(self, indices, outputSize, output));
  auto output_size = op_infer::max_pool2d_out_size(self, outputSize);
  npu_preparation::check_tensor({self, indices}, output, self.scalar_type(), output_size);

  EXEC_NPU_CMD(aclnnMaxUnpool2d, self, indices, outputSize, output);
  return output;
};

at::Tensor max_unpool2d(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnMaxUnpool2d, acl_op::max_unpool2d(self, indices, output_size));
  auto outputSize = op_infer::max_pool2d_out_size(self, output_size);
  at::Tensor output = npu_preparation::apply_tensor_without_format(self, outputSize);
  op_api::max_unpool2d_out(self, indices, output_size, output);
  return output;
}
} // namespace op_api
