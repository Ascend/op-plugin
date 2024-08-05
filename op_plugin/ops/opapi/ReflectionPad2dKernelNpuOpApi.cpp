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

at::Tensor& reflection_pad2d_out(const at::Tensor& self,
                                 at::IntArrayRef padding,
                                 at::Tensor& out) {
  DO_COMPATIBILITY(aclnnReflectionPad2d, acl_op::reflection_pad2d_out(self, padding, out));
  auto output_size = op_infer::reflection_pad2d_npu_out_size(self, padding);
  npu_preparation::check_tensor({self}, out, self, output_size);
  EXEC_NPU_CMD(aclnnReflectionPad2d, self, padding, out);
  return out;
}

at::Tensor reflection_pad2d(const at::Tensor& self,
                            at::IntArrayRef padding) {
  DO_COMPATIBILITY(aclnnReflectionPad2d, acl_op::reflection_pad2d(self, padding));
  auto output_size = op_infer::reflection_pad2d_npu_out_size(self, padding);
  at::Tensor out = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnReflectionPad2d, self, padding, out);
  return out;
}

}
