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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace{
inline at::Tensor all_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dimList,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ReduceAll")
      .Input(self)
      .Input(dimList, at::kLong)
      .Output(result) 
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}
} // namespace

at::Tensor& all_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> dimList = {dim};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dimList, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    all_out_npu_nocheck(contiguous_result, self, dimList, keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    all_out_npu_nocheck(result, self, dimList, keepdim);
  }
  return result;
}

at::Tensor all(const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == at::kBool || self.scalar_type() == at::kByte,
      "all only supports torch.uint8 and torch.bool dtypes");
  TORCH_CHECK(dim >= -(self.dim()) && dim < self.dim(),
      "The value of dim must be greater than or equal to -self.dim() and less than self.dim()");
  if (self.numel() == 0) {
    auto output_size = op_infer::infersize_all(self, dim);
    at::Tensor result = npu_preparation::ApplyTensor(
        output_size,
        self.options().dtype(at::kInt),
        self);
    op_plugin::fill_(result, 1);
    result = op_plugin::npu_dtype_cast(result, at::kBool);
    return result;
  }
  at::IntArrayRef dims(dim);
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  all_out_npu_nocheck(result, self, {dim}, keepdim);
  return result;
}

at::Tensor all(const at::Tensor& self) {
  TORCH_CHECK(self.scalar_type() == at::kBool || self.scalar_type() == at::kByte,
      "all only supports torch.uint8 and torch.bool dtypes");
  if (self.numel() == 0) {
    at::Tensor result = npu_preparation::ApplyTensor({}, self.options().dtype(at::kInt), self);
    op_plugin::fill_(result, 1);
    result = op_plugin::npu_dtype_cast(result, at::kBool);
    return result;
  }

  at::IntArrayRef dims;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  all_out_npu_nocheck(
      result,
      self,
      calcu_op_util::GetDimlistForTensor(self),
      false);
  return result;
}
} // namespace op_plugin
