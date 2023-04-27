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
using npu_compile_type = at_npu::native::CompileType;

at::Tensor one_hot(const at::Tensor& self, int64_t num_classes) {
  at::Scalar on_value = 1;
  at::Scalar off_value = 0;
  int64_t axis = -1;
  int64_t depth;
  auto self_temp = op_plugin::npu_dtype_cast(self, at::kFloat);

  TORCH_CHECK(self_temp.dim() < 8, "NPU error,can not support the input tensor's dim bigger than 7.");
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      depth = num_classes;
    }
  }

  TORCH_CHECK(self_temp.min().item().toLong() >= 0, "Class values must be non-negative.");
  if (num_classes == -1) {
    depth = self_temp.max().item().toLong() + 1;
  } else {
    TORCH_CHECK(
        num_classes > self_temp.max().item().toLong(),
        "Class values must be smaller than num_classes.");
    depth = num_classes;
  }

  auto output_size = op_infer::array_to_small_vector(self.sizes());
  output_size.emplace_back(depth);
  at::Tensor result = npu_preparation::ApplyTensor(output_size, self.options().dtype(at::ScalarType::Int), self);
  at::Scalar depth_copy = depth;
  at_npu::native::OpCommand cmd;
  cmd.Name("OneHot")
      .Input(self)
      .Input(depth_copy, at::ScalarType::Int, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(on_value, at::ScalarType::Int)
      .Input(off_value, at::ScalarType::Int)
      .Output(result)
      .Attr("axis", axis)
      .Run();
  return result;
}
} // namespace op_plugin