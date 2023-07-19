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
at::Tensor& scatter_npu_nocheck(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ScatterElements")
      .Input(self)
      .Input(index)
      .Input(src)
      .Output(self)
      .Attr("axis", dim)
      .Run();
  return self;
}

at::Tensor& scatter_npu_src_impl(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Tensor& src_ex) {
  at::ScalarType self_type = self.scalar_type();
  if (self_type == at::ScalarType::Half) {
    self = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::Tensor index(index_ex);
  if (index.scalar_type() == at::ScalarType::Half) {
    index = op_plugin::npu_dtype_cast(index, at::ScalarType::Float);
  }

  at::Tensor src(src_ex);
  if (src.scalar_type() != self.scalar_type()) {
    src = op_plugin::npu_dtype_cast(src, self.scalar_type());
  }

  scatter_npu_nocheck(self, dim, index, src);
  
  if(self.scalar_type() != self_type) {
    self = op_plugin::npu_dtype_cast(self, self_type);
  }

  return self;
}
} // namespace

at::Tensor& scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self, src, index},
      result,
      self);
  result = at_npu::native::NPUNativeFunctions::copy_(result, self, false);
  scatter_npu_src_impl(result, dim, index, src);
  return result;
}

at::Tensor& scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    at::Tensor& result) {
  at::Tensor src_tensor = scalar_to_tensor(value).to(at::ScalarType::Float);
  src_tensor = calcu_op_util::CopyTensorHostToDevice(src_tensor);
  at::Tensor src_tensor_broadcast = op_plugin::npu_broadcast(
      src_tensor, op_infer::array_to_small_vector(index.sizes()));
  npu_preparation::CheckOut(
      {self, index, src_tensor_broadcast},
      result,
      self);
  result = at_npu::native::NPUNativeFunctions::copy_(result, self, false);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    scatter_npu_src_impl(contiguous_result, dim, index, src_tensor_broadcast);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    scatter_npu_src_impl(result, dim, index, src_tensor_broadcast);
  }
  scatter_npu_src_impl(result, dim, index, src_tensor_broadcast);
  return result;
}
} // namespace op_plugin