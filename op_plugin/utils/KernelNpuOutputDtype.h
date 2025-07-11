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

#ifndef OP_PLUGIN_UTILS_KERNEL_NPU_INFER_DTYPE
#define OP_PLUGIN_UTILS_KERNEL_NPU_INFER_DTYPE

#include <ATen/ATen.h>

#include "op_plugin/utils/Export.h"

namespace op_infer {

OP_PLUGIN_HIDDEN at::ScalarType angle_out_dtype(const at::Tensor& self);
OP_PLUGIN_HIDDEN at::ScalarType polar_out_dtype(const at::Tensor& abs, const at::Tensor& angle);
OP_PLUGIN_HIDDEN at::ScalarType npu_group_quant_dst_type(c10::optional<at::ScalarType> dst_dtype);
OP_PLUGIN_HIDDEN at::ScalarType clamp_out_dtype(const at::Tensor& self, const c10::optional<at::Tensor>& min, const c10::optional<at::Tensor>& max);
OP_PLUGIN_HIDDEN at::ScalarType clamp_scalar_out_dtype(const at::Tensor& self, const c10::optional<at::Scalar>& min, const c10::optional<at::Scalar>& max);

} // namespace op_infer
#endif // OP_PLUGIN_UTILS_KERNEL_NPU_INFER_DTYPE
