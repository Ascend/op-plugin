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

at::Tensor median(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnMedian, acl_op::median(self));
  at::SmallVector<int64_t, op_infer::SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnMedian, self, result);
  return result;
}

std::tuple<at::Tensor, at::Tensor> median(const at::Tensor& self,
                                          int64_t dim,
                                          bool keepdim) {
  DO_COMPATIBILITY(aclnnMedianDim, acl_op::median(self, dim, keepdim));
  at::SmallVector<int64_t, op_infer::SIZE> dims = {dim};
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor values = npu_preparation::apply_tensor_without_format(self, outputSize);
  at::Tensor indices = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kLong));
  EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor, at::Tensor> median(const at::Tensor& self,
                                          at::Dimname dim,
                                          bool keepdim) {
  DO_COMPATIBILITY(aclnnMedianDim, acl_op::median(self, dim, keepdim));
  int64_t real_dim = dimname_to_position(self, dim);
  at::SmallVector<int64_t, op_infer::SIZE> dims = {real_dim};
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor values = npu_preparation::apply_tensor_without_format(self, outputSize);
  at::Tensor indices = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kLong));
  EXEC_NPU_CMD(aclnnMedianDim, self, real_dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> median_out(const at::Tensor& self,
                                                int64_t dim,
                                                bool keepdim,
                                                at::Tensor& values,
                                                at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMedianDim, acl_op::median_out(self, dim, keepdim, values, indices));
  at::SmallVector<int64_t, op_infer::SIZE> dims = {dim};
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  npu_preparation::check_tensor({self}, values, values.scalar_type(), outputSize);
  npu_preparation::check_tensor({self}, indices, indices.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> median_out(const at::Tensor& self,
                                                at::Dimname dim,
                                                bool keepdim,
                                                at::Tensor& values,
                                                at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnMedianDim, acl_op::median_out(self, dim, keepdim, values, indices));
  int64_t real_dim = dimname_to_position(self, dim);
  at::SmallVector<int64_t, op_infer::SIZE> dims = {real_dim};
  auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  npu_preparation::check_tensor({self}, values, values.scalar_type(), outputSize);
  npu_preparation::check_tensor({self}, indices, indices.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnMedianDim, self, real_dim, keepdim, values, indices);
  return std::tie(values, indices);
}

}  // namespace op_api

