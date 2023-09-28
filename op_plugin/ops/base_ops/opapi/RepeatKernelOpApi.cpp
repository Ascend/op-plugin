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

at::Tensor repeat(const at::Tensor &self, at::IntArrayRef repeats)
{
  DO_COMPATIBILITY(aclnnRepeat, acl_op::repeat(self, repeats));
  auto outputSize = op_infer::repeat_npu_output_size(self, repeats);
  at::Tensor result = npu_preparation::apply_tensor_with_sizes(outputSize, self.options());
  EXEC_NPU_CMD(aclnnRepeat, self, repeats, result);
  return result;
}
}  // namespace op_api

