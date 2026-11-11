// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <ATen/ops/_nested_tensor_from_mask_left_aligned_native.h>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

// Probe whether the padding mask is left-aligned (True == valid, each row valid
// prefix is contiguous, so the upper-level SDPA can take the nested-tensor fast
// path). Delegates to the ATen native implementation in NestedTensorMath.cpp,
// which is composed of basic tensor ops (zeros/cat/argmin/cumsum/select/equal)
// and therefore runs on NPU tensors directly through the dispatcher.
bool _nested_tensor_from_mask_left_aligned(const at::Tensor& t, const at::Tensor& mask) {
    return at::native::NestedTensor_nested_tensor_from_mask_left_aligned(t, mask);
}

}  // namespace op_api
