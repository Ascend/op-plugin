// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#ifndef OP_PLUGIN_UTILS_KERNEL_NPU_NEW_PARAMS
#define OP_PLUGIN_UTILS_KERNEL_NPU_NEW_PARAMS

#include <ATen/ATen.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/Export.h"

namespace op_infer {

OP_PLUGIN_HIDDEN int64_t npu_gelu_approximate_mode(c10::string_view approximate);
OP_PLUGIN_HIDDEN std::string npu_gelu_approximate_str(c10::string_view approximate);
OP_PLUGIN_HIDDEN bool npu_add_rms_norm_quant_param_check(c10::optional<at::Tensor> scales2,
                                                         c10::optional<at::Tensor> zero_points2,
                                                         int64_t axis,
                                                         bool div_mode);

} // namespace op_infer
#endif // OP_PLUGIN_UTILS_KERNEL_NPU_NEW_PARAMS
