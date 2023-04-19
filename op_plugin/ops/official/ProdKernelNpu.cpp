// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
// See the License for the specific

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
static inline int64_t calculate_prod_output_format(
    const at::Tensor& self,
    at::IntArrayRef size) {
  int64_t npu_format = calcu_op_util::GetTensorNpuFormat(self);
  // scalar scene no support nz
  if (size.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  return npu_format;
}

at::Tensor& prod_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, at_npu::native::N> dimList,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  at_npu::native::OpCommand cmd;
    cmd.Name("ReduceProd")
    .Input(self)
    .Input(dimList)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}
} //namespace

at::Tensor& prod_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  at::Tensor self_tmp = self;
  // fp16 transform：fp32 for precise
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  auto output_size = op_infer::prod_npu_output_size(self, dim, keepdim);
  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  npu_preparation::CheckOut(
      {self_tmp},
      result,
      ACL_FORMAT_ND,
      dstType,
      output_size);

  at::Tensor result_tmp = result;
  if (result_tmp.scalar_type() == at::ScalarType::Half) {
    result_tmp = op_plugin::npu_dtype_cast(result_tmp, at::ScalarType::Float);
  }

  if (!npu_utils::check_match(&result_tmp)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_tmp);
    prod_out_npu_nocheck(contiguous_result, self_tmp, {dim}, keepdim, dtype);
    npu_utils::format_fresh_view(result_tmp, contiguous_result);
  } else {
    prod_out_npu_nocheck(result_tmp, self_tmp, {dim}, keepdim, dtype);
  }

  if (result_tmp.scalar_type() != dstType) {
    result_tmp = op_plugin::npu_dtype_cast(result_tmp, dstType);
  }
  result.copy_(result_tmp);

  return result;
}

at::Tensor& prod_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype,
    at::Tensor& result) {
  return prod_out(
      self, dimname_to_position(self, dim), keepdim, dtype, result);
}

at::Tensor prod(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  at::Tensor self_tmp = self;
  // Input transform：fp16 to fp32
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  auto output_size = op_infer::prod_npu_output_size(self_tmp, dim, keepdim);

  int64_t npu_format = calculate_prod_output_format(self_tmp, output_size);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size, self_tmp.options(), npu_format);

  prod_out_npu_nocheck(result, self_tmp, {dim}, keepdim, dtype);

  if (result.scalar_type() != dstType) {
    result = op_plugin::npu_dtype_cast(result, dstType);
  }

  return result;
}

at::Tensor prod(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    c10::optional<at::ScalarType> dtype) {
  return op_plugin::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

at::Tensor prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  at::Tensor self_tmp = self;
  // Input transform：fp16 to fp32
  if (self.scalar_type() == at::ScalarType::Half) {
    self_tmp = op_plugin::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

  auto output_size = op_infer::prod_npu_output_size(self, false);

  int64_t npu_format = calculate_prod_output_format(self, output_size);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(
      output_size, self_tmp.options(), npu_format);

  prod_out_npu_nocheck(
      result, self_tmp, calcu_op_util::GetDimlistForTensor(self), false, dtype);

  if (result.scalar_type() != dstType) {
    result = npu_dtype_cast(result, dstType);
  }

  return result;
}
} // namespace op_plugin
