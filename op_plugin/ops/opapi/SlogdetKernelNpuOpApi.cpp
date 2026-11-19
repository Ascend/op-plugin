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

std::tuple<at::Tensor, at::Tensor> slogdet(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSlogdet, acl_op::slogdet(self));
  // input dimension at least 2
  TORCH_CHECK(
      self.ndimension() >= 2,
      "Expected nonempty least 2D tensor, but got a tensor with sizes ",
      self.dim(),
      OPS_ERROR(ErrCode::PARAM));
  // calculate the output size
  auto outputSize = op_infer::array_to_small_vector(self.sizes());
  outputSize.erase(outputSize.end() - 2, outputSize.end());
  // construct the output tensor of the NPU
  at::Tensor sign = npu_preparation::apply_tensor(self, outputSize);
  // aclnnSlogdet requires logOut to keep the input dtype for complex input,
  // while torch.slogdet returns a real-valued logabsdet, so compute in the
  // input dtype first and take the real part afterwards.
  at::Tensor log = npu_preparation::apply_tensor(self, outputSize);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

  if (!self.is_complex()) {
    return std::tie(sign, log);
  }
  at::Tensor logabsdet = npu_preparation::apply_tensor_without_format(
      outputSize, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  logabsdet.copy_(at::real(log));
  return std::tie(sign, logabsdet);
}
} // namespace op_api
