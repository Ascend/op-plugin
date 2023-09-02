// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
std::tuple<at::Tensor, at::Tensor> prelu_backward(const at::Tensor& grad_output,
                                                                      const at::Tensor& self,
                                                                      const at::Tensor& weight) {
  DO_COMPATIBILITY(aclnnPreluBackward, acl_op::prelu_backward(grad_output, self, weight));
  c10::SmallVector<int64_t, SIZE> output_size = op_infer::prelu_backward_npu_grad_weight_output_size(weight);

  at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output);
  at::Tensor grad_weight = npu_preparation::apply_tensor_without_format(weight, output_size);
  EXEC_NPU_CMD(aclnnPreluBackward, grad_output, self, weight, grad_input, grad_weight);
  return std::tie(grad_input, grad_weight) ;
}

} // namespace op_api
