// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2020-2026 Intel Corporation
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/native/Resize.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const static int64_t LEN_MIN = 32;
const static int64_t LEN_MAX = 64;
const static int64_t DIM_2 = 2;
const static int64_t DIM_MIN_16 = 16;
const static int64_t DIM_MIN_8 = 8;
const static int64_t DIM_MAX = 65536;

bool is_fused_op_optim(const at::Tensor& self, at::IntArrayRef split_sizes) {
  if (!op_plugin::utils::is_gte_cann_version_830rc1()) {
    return false;
  }

  if (self.dim() != DIM_2) {
    return false;
  }

  int64_t len = split_sizes.size();
  if ((len <= LEN_MIN) || (len > LEN_MAX)) {
    return false;
  }

  at::ScalarType dtype = self.scalar_type();

  int64_t dim0 = self.size(0);
  int64_t dim1 = self.size(1);

  if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
    return (dim0 <= DIM_MIN_16) && (dim1 > DIM_MAX);
  } else if (dtype == at::ScalarType::Float) {
    return (dim0 <= DIM_MIN_8) && (dim1 > DIM_MAX);
  } else {
    return false;
  }
}

void split_with_sizes_copy_out(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  if (is_fused_op_optim(self, split_sizes) ||
      (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950 &&
       op_plugin::utils::is_gte_cann_version_920())) {
    if (dim < 0) {
      dim = at::maybe_wrap_dim(dim, self.dim());
    }
    TORCH_CHECK(
      self.dim() != 0, "split expects at least a 1-dimensional tensor")
    const int64_t dim_size = self.size(dim);
    int64_t split_sizes_sum = 0;
    for (const auto i : c10::irange(split_sizes.size())) {
      TORCH_CHECK(
          split_sizes[i] >= 0,
          "split_with_sizes expects split_sizes have only non-negative ",
          "entries, but got split_sizes=",
          split_sizes[i]);
      split_sizes_sum += split_sizes[i];
    }
    TORCH_CHECK(
        split_sizes_sum == dim_size,
        "split_with_sizes expects split_sizes to sum exactly to ",
        dim_size,
        " (input tensor's size at dimension ",
        dim,
        "), ",
        "but got split_sizes=",
        split_sizes);

    TORCH_CHECK(
        out.size() == split_sizes.size(),
        "split_with_sizes_copy_out() expected an out= argument of size ",
        split_sizes.size(),
        ", got size ",
        out.size());
    auto out_shape = self.sizes().vec();
    for (const auto i : c10::irange(split_sizes.size())) {
      out_shape[dim] = split_sizes[i];
      if (at::native::resize_output_check(out[i], out_shape)) {
        out[i].resize_(out_shape);
      }
      TORCH_CHECK(
          out[i].dtype() == self.dtype(),
          "Expected out tensor to have dtype ",
          self.dtype(),
          ", but got ",
          out[i].dtype(),
          " instead");
      TORCH_CHECK(
          out[i].device() == self.device(),
          "Expected out tensor to have device ",
          self.device(),
          ", but got ",
          out[i].device(),
          " instead");
    }
    EXEC_NPU_CMD(aclnnSplitWithSize, self, split_sizes, dim, out);
  } else {
    at::native::split_with_sizes_copy_out(self, split_sizes, dim, out);
  }
}

} // namespace op_api
