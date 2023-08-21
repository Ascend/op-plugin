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
using npu_preparation = at_npu::native::OpPreparation;

static std::tuple<c10::SmallVector<int64_t, op_infer::N>, c10::SmallVector<int64_t, op_infer::N>> qr_infer_shape(
    const at::Tensor& self,
    bool some) {
  int m = self.size(-2);
  int n = self.size(-1);
  auto k = std::min<int>(m, n);
  auto shape = op_infer::array_to_small_vector(self.sizes());
  c10::SmallVector<int64_t, op_infer::N> Qsize(shape.begin(), shape.end()-2);
  c10::SmallVector<int64_t, op_infer::N> Rsize(shape.begin(), shape.end()-2);
  if (some) {
    Qsize.insert(Qsize.end(), {m, k});
    Rsize.insert(Rsize.end(), {k, n});
  } else {
    Qsize.insert(Qsize.end(), {m, m});
    Rsize.insert(Rsize.end(), {m, n});
  }
  return std::tie(Qsize, Rsize);
}

static inline void qr_check(const at::Tensor& self) {
  TORCH_CHECK(self.ndimension() >= 2, "Expected nonempty least 2D tensor, but got a tensor with sizes ", self.dim());
}

std::tuple<at::Tensor&, at::Tensor&> qr_out(
    const at::Tensor& self,
    bool some,
    at::Tensor& Q,
    at::Tensor& R) {
  DO_COMPATIBILITY(aclnnQr, acl_op::qr_out(self, some, Q, R));
  qr_check(self);
  auto sizes = qr_infer_shape(self, some);
  npu_preparation::check_tensor(
      {self},
      Q,
      self,
      std::get<0>(sizes));
  npu_preparation::check_tensor(
      {self},
      R,
      self,
      std::get<1>(sizes));
  EXEC_NPU_CMD(aclnnQr, self, some, Q, R);
  return std::tie(Q, R);
}

std::tuple<at::Tensor, at::Tensor> qr(const at::Tensor& self, bool some) {
  DO_COMPATIBILITY(aclnnQr, acl_op::qr(self, some));
  qr_check(self);
  auto sizes = qr_infer_shape(self, some);
  at::Tensor Q = npu_preparation::apply_tensor_without_format(std::get<0>(sizes), self.options());
  at::Tensor R = npu_preparation::apply_tensor_without_format(std::get<1>(sizes), self.options());
  EXEC_NPU_CMD(aclnnQr, self, some, Q, R);
  return std::tie(Q, R);
}

}  // namespace op_api

