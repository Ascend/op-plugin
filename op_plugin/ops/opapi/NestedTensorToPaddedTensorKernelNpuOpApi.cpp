// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include <ATen/native/nested/NestedTensorMath.h>

namespace op_api {
at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    at::OptionalIntArrayRef output_size) {
  // 调用 PyTorch 原生 CPU 实现
  return at::native::NestedTensor_to_padded_tensor_generic(
      self, padding, output_size);
}
} // namespace op_api
