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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& fill_out_nocheck(at::Tensor& result, at::Tensor& self, const at::Tensor& other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& fill_out_nocheck(at::Tensor& result, at::Tensor& self, at::Scalar other) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(other, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& fill_out_nocheck(at::Tensor& self, const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    fill_out_nocheck(self, self, other.item());
  } else {
    fill_out_nocheck(self, self, other);
  }
  return self;
}
} // namespace

at::Tensor& fill_(at::Tensor& self, const at::Tensor& other) {
  auto other_dim = other.dim();
  TORCH_CHECK(other_dim <= 1, "fill_ only supports 0 or 1 dimension value tensor but got tensor with ",
      other_dim, " dimension.");
  npu_preparation::CheckMemory({self, other}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    fill_out_nocheck(contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    fill_out_nocheck(self, other);
  }
  return self;
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& other) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    fill_out_nocheck(contiguous_self, contiguous_self, other);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    fill_out_nocheck(self, self, other);
  }
  return self;
}
}  // namespace op_plugin
