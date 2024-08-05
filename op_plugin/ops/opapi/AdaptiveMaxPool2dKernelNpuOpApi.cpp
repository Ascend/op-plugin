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

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& output,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d,
                   acl_op::adaptive_max_pool2d_out(self, output_size, output, indices));
  npu_preparation::check_memory({self}, {output, indices});

  auto outputSize = op_infer::max_pool2d_out_size(self, output_size);
  npu_preparation::check_tensor({self}, output, self.scalar_type(), outputSize);
  npu_preparation::check_tensor({self}, indices, at::ScalarType::Long, outputSize);

  EXEC_NPU_CMD(aclnnAdaptiveMaxPool2d, self, output_size, output, indices);
  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d, acl_op::adaptive_max_pool2d(self, output_size));
  auto outputSize = op_infer::max_pool2d_out_size(self, output_size);
  at::Tensor output = npu_preparation::apply_tensor_without_format(self, outputSize);
  at::Tensor indices = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kLong));

  op_api::adaptive_max_pool2d_out(self, output_size, output, indices);
  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}
}
