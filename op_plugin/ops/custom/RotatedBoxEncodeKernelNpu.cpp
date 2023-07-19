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

at::Tensor npu_rotated_box_encode(
    const at::Tensor& self, 
    const at::Tensor& gtBox, 
    const at::Tensor& weight){
  at::Tensor result = npu_preparation::ApplyTensor(self);
  at::Tensor weight_cpu = weight.to(at::Device(at::kCPU), at::kFloat);
  at::ArrayRef<float> weight_list(weight_cpu.data_ptr<float>(), weight_cpu.numel());

  at_npu::native::OpCommand cmd;
  cmd.Name("RotatedBoxEncode")
      .Input(self)
      .Input(gtBox)
      .Output(result)
      .Attr("weight", weight_list)
      .Run();
  return result;
}
} // namespace op_plugin