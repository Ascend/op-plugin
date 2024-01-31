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

std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor &self,
                                            const int64_t dim,
                                            bool keepdim) {
  DO_COMPATIBILITY(aclnnAminmaxDim, acl_op::_aminmax(self, dim, keepdim));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  auto min = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  auto max = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxDim, self, dim, keepdim, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> _aminmax(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnAminmaxDim, acl_op::_aminmax(self));
  c10::SmallVector<int64_t, N> dimlist;
  dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
  at::IntArrayRef dims = dimlist;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  auto min = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  auto max = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxAll, self, min, max);
  return std::tie(min, max);
}

at::Tensor& amin_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAmin, acl_op::amin_out(self, dim, keepdim, result));

  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  // check result for return
  at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, result);
  return result;
}

at::Tensor amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnAmin, acl_op::amin(self, dim, keepdim));

  // calculate the output size
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);

  // construct the output tensor of the NPU
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self, outputSize);
  EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, result);
  return result;
}

} // namespace op_api
