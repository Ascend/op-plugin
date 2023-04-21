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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& index_select_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  if (self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE(
        "The oprator of index_select is executed, Currently High Accuracy but Low Performance OP with 64-bit has been "
        "used, Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  c10::SmallVector<int64_t, N> dim_vec = {dim};
  int64_t batch_dims = 0;
  at_npu::native::OpCommand cmd;
  cmd.Name("GatherV2")
      .Input(self)
      .Input(index)
      .Input(dim_vec)
      .Output(result)
      .Attr("batch_dims", batch_dims)
      .Run();
  return result;
}
} // namespace

at::Tensor& index_select_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& result) {
  at::Tensor index_tmp(index);
  if (index_tmp.ndimension() == 0) {
    index_tmp = index.unsqueeze(0);
  }
  auto output_size = op_infer::index_select_npu_output_size(self, dim, index_tmp);
  int64_t npu_format = calcu_op_util::GetTensorNpuFormat(self);
  if (output_size.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = op_plugin::npu_dtype_cast(input, at::kInt);
  }
  npu_preparation::CheckOut(
      {input, index_tmp},
      result,
      npu_format,
      input.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    index_select_out_npu_nocheck(contiguous_result, input, dim, index_tmp);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    index_select_out_npu_nocheck(result, input, dim, index_tmp);
  }

  if (self.dtype() == at::kBool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  at::Tensor index_tmp(index);
  if (index_tmp.ndimension() == 0) {
    index_tmp = index.unsqueeze(0);
  }
  auto output_size = op_infer::index_select_npu_output_size(self, dim, index_tmp);
  int64_t npu_format = calcu_op_util::GetTensorNpuFormat(self);
  if (output_size.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = op_plugin::npu_dtype_cast(input, at::kInt);
  }
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(input, output_size, npu_format);
  index_select_out_npu_nocheck(result, input, dim, index_tmp);
  if (self.dtype() == at::kBool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

at::Tensor& index_select_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    at::Tensor& result) {
  at::Tensor index_tmp(index);
  if (index_tmp.ndimension() == 0) {
    index_tmp = index.unsqueeze(0);
  }
  return op_plugin::index_select_out(self, dimname_to_position(self, dim), index_tmp, result);
}

at::Tensor index_select(
    const at::Tensor& self, 
    at::Dimname dim, 
    const at::Tensor& index) {
  return op_plugin::index_select(self, dimname_to_position(self, dim), index);
}
} // namespace op_plugin
