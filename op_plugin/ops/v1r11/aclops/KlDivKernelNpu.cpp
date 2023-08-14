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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
  at::Tensor result = reduction == at::Reduction::None ?
      npu_preparation::apply_tensor(self) : npu_preparation::apply_tensor({}, self.options(), self);
  std::string reduction_str;
  if (reduction == at::Reduction::Mean) {
    reduction_str = "batchmean";
  } else if (reduction == at::Reduction::Sum) {
    reduction_str = "sum";
  } else if (reduction == at::Reduction::None) {
    reduction_str = "none";
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("KLDiv")
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Attr("log_target", log_target)
      .Run();
  if (reduction == at::Reduction::Mean) {
    auto input_shape = self.sizes();
    int batch_square_size = c10::multiply_integers(input_shape) / input_shape[0];
    result.div_(batch_square_size);
  }
  return result;
}
} // namespace op_plugin
