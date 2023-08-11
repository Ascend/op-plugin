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

namespace {
at::Tensor& repeat_interleave_out_nocheck(at::Tensor& result, const at::Tensor& self, int64_t repeats) {
  at::Scalar repeat = repeats;
  at_npu::native::OpCommand cmd;
  cmd.Name("RepeatInterleave")
      .Input(self)
      .Input(repeat, at::kLong)
      .Output(result)
      .Attr("axis", (int64_t)0)
      .Run();

  return result;
}

at::Tensor& repeat_interleave_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& repeats) {
  at_npu::native::OpCommand cmd;
  cmd.Name("RepeatInterleave")
      .Input(self)
      .Input(repeats)
      .Output(result)
      .Attr("axis", (int64_t)0)
      .Run();

  return result;
}

at::Tensor repeat_interleave_without_symint(
    const at::Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t real_dim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (real_dim >= -self_dim) && (real_dim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  TORCH_CHECK(
      repeats >= 1,
      "repeats can not be negative.");
  at::Tensor self_tensor = self;
  if (!dim.has_value()) {
    self_tensor = at::flatten(self_tensor);
  }
  if (repeats == 1) {
    return self_tensor;
  }

  if (self_dim > 1 && real_dim != 0) {
    self_tensor = self_tensor.transpose(0, real_dim);
  }

  auto op_infer_output_size = op_infer::repeat_interleave_npu_output_size(self_tensor, repeats, 0);
  at::Tensor result = npu_preparation::ApplyTensorWithFormat(self_tensor, op_infer_output_size, ACL_FORMAT_ND);
  repeat_interleave_out_nocheck(result, self_tensor, repeats);
  if (self_dim > 1 && real_dim != 0) {
    result = result.transpose(0, real_dim);
  }
  return result;
}
} // namespace

at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t repeats_val = repeats.guard_int(__FILE__, __LINE__);
  return repeat_interleave_without_symint(self, repeats_val, dim, output_size);
}

at::Tensor repeat_interleave(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t real_dim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (real_dim >= -self_dim) && (real_dim <= self_dim - 1),
      "dim value should be in the range of [", -self_dim, ",", self_dim - 1, "].");

  at::Tensor self_tensor = self;
  at::Tensor repeats_tensor = repeats;
  if (!dim.has_value()) {
    self_tensor = at::flatten(self_tensor);
  }
  if (repeats.dim() == 1 && repeats.size(0) == 1) {
    return self_tensor;
  }

  TORCH_CHECK(
      repeats.size(0) == self_tensor.size(real_dim),
      "repeats must have the same size as input along dim.");

  if (self_dim > 1 && real_dim != 0) {
    self_tensor = self_tensor.transpose(0, real_dim);
  }

  repeats_tensor = op_plugin::npu_dtype_cast(repeats_tensor, at::ScalarType::Int);
  repeats_tensor = op_plugin::npu_dtype_cast(repeats_tensor, at::ScalarType::Float);
  auto op_infer_output_size = op_infer::repeat_interleave_npu_output_size(self_tensor, repeats_tensor, 0);

  at::Tensor result = npu_preparation::ApplyTensorWithFormat(self_tensor, op_infer_output_size, ACL_FORMAT_ND);
  repeat_interleave_out_nocheck(result, self_tensor, repeats);
  if (self_dim > 1 && real_dim != 0) {
    result = result.transpose(0, real_dim);
  }
  return result;
}
} // namespace op_plugin
