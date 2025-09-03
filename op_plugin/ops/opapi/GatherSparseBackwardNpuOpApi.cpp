// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/TensorSubclassLikeUtils.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_gather_sparse_index_backward_symint(const at::Tensor& grad, c10::SymIntArrayRef self_sizes, const at::Tensor& index)
{
    at::Tensor output = grad.new_zeros_symint(self_sizes, grad.options());
    if (at::isTensorSubclassLike(index)) {
        return at_npu::native::custom_ops::_npu_index_add(output, index, grad);
    }
    return at_npu::native::custom_ops::_npu_index_add_(output, index, grad);
}
}
