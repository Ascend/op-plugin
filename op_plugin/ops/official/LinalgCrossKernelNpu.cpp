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

namespace {
at::Tensor linalg_cross_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  return is_self_wrapped ? other : self;
}

at::Tensor& linalg_cross_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<int64_t> dim) {
  int64_t real_dim = dim.has_value() ? dim.value() : -65530;
  at_npu::native::OpCommand cmd;
  cmd.Name("Cross")
      .Input(self)
      .Input(other)
      .Output(result)
      .Attr("dim", real_dim)
      .Run();
  return result;
}
} // namespace

at::Tensor& linalg_cross_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim,
    at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor output_tensor = linalg_cross_dest_output(self, other);
  npu_preparation::CheckOut(
      {self},
      result,
      calcu_op_util::GetTensorNpuFormat(output_tensor),
      self.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    linalg_cross_out_npu_nocheck(contiguous_result, self, other, dim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    linalg_cross_out_npu_nocheck(result, self, other, dim);
  }
  return result;
}

at::Tensor linalg_cross(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor output_tensor = linalg_cross_dest_output(self, other);
  at::Tensor result = npu_preparation::ApplyTensor(output_size, self.options(), output_tensor);
  linalg_cross_out_npu_nocheck(result, self, other, dim);
  return result;
}
} // namespace op_plugin