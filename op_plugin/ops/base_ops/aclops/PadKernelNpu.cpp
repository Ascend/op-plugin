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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_pad(const at::Tensor& input, at::IntArrayRef paddings) {
  auto output_size = op_infer::pad_npu_output_size(input, paddings);
  at::Tensor output = npu_preparation::ApplyTensor(input, output_size);  
  c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(paddings);
  paddings_vector.resize(2 * input.dim(), 0);

  at_npu::native::OpCommand cmd;
  cmd.Name("Pad")
      .Input(input)
      .Input(paddings_vector)
      .Output(output)
      .Run();

  return output;
}
} // namespace acl_op
