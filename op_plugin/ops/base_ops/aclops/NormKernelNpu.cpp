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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
float calculate_p(c10::optional<at::Scalar> p) {
  if (p.has_value()) {
    float val = op_plugin::utils::get_scalar_float_value(p.value());
    if (val == INFINITY) {
      return static_cast<float>(INT_MAX); // p = inf
    } else if (val == -INFINITY) {
      return static_cast<float>(INT_MIN); // p = -inf
    } else {
      return p.value().toFloat();
    }
  } else {
    return static_cast<float>(2.0); // default: p = 2.0
  }
}

at::Tensor& norm_out_npu_nocheck(
    at::Tensor& out,
    const at::Tensor& self,
    c10::optional<at::Scalar> p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype) {
  at::Tensor fp32_self(self);
  if (self.scalar_type() != at::ScalarType::Float) {
    fp32_self = at_npu::native::custom_ops::npu_dtype_cast(fp32_self, at::ScalarType::Float);
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(fp32_self, dim, keepdim);
  at::Tensor result_temp = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
  at::Tensor result = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
  auto pvalue = calculate_p(p);
  at_npu::native::OpCommand cmd1;
  cmd1.Name("LpNormReduceV2")
      .Input(fp32_self)
      .Output(result_temp)
      .Attr("p", pvalue)
      .Attr("axes", dim)
      .Attr("keepdim", keepdim)
      .Attr("epsilon", static_cast<float>(0))
      .Run();

  at_npu::native::OpCommand cmd2;
  cmd2.Name("LpNormUpdateV2")
      .Input(result_temp)
      .Output(result)
      .Attr("p", pvalue)
      .Attr("epsilon", static_cast<float>(0))
      .Run();
  if (result.scalar_type() != dtype) {
    result = at_npu::native::custom_ops::npu_dtype_cast(result, dtype);
  }
  // until now, can not support resize shape of out correctly,
  // so the shape of out must be equal to output_size
  out = out.copy_(result);
  return out;
}
} // namespace

at::Tensor& norm_out(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    norm_out_npu_nocheck(contiguous_result, self, p, dim, keepdim, self.scalar_type());
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    norm_out_npu_nocheck(result, self, p, dim, keepdim, self.scalar_type());
  }
  return result;
}

at::Tensor& norm_out(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      dtype,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    norm_out_npu_nocheck(contiguous_result, self, p, dim, keepdim, dtype);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    norm_out_npu_nocheck(result, self, p, dim, keepdim, dtype);
  }

  return result;
}

at::Tensor norm(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype) {
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  at::Tensor out = npu_preparation::ApplyTensorWithSizes(output_size, self.options().dtype(dtype));
  norm_out_npu_nocheck(out, self, p, dim, keepdim, dtype);
  return out;
}

at::Tensor norm(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::ScalarType dtype) {
  return acl_op::norm(self, p, {}, false, dtype);
}

at::Tensor norm(
    const at::Tensor& self,
    const at::Scalar& p) {
  return acl_op::norm(self, p, {}, false, self.scalar_type());
}

at::Tensor norm(
    const at::Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim) {
  return acl_op::norm(self, p, dim, keepdim, self.scalar_type());
}

} // namespace acl_op
