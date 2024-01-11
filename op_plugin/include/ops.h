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

#ifndef OP_PULGIN_INCLUDE_OPS
#define OP_PULGIN_INCLUDE_OPS

#include <ATen/ATen.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace at_npu {
namespace native {
TORCH_NPU_API at::Tensor npu_dropout_gen_mask(const at::Tensor &self, at::IntArrayRef size, double p, int64_t seed,
                                              int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync);
}
}  // namespace at_npu

#endif  // OP_PULGIN_INCLUDE_OPS
