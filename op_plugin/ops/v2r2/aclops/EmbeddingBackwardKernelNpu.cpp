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

#include "op_plugin/AclOpsInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"


namespace acl_op {
at::Tensor embedding_backward_symint(
    const at::Tensor& grad, 
    const at::Tensor& indices, 
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq, 
    bool sparse) {
    TORCH_CHECK(sparse == false, "the current NPU does not yet support sparse tensor, when sparse is set to True" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    // run dense tensor backward
    return at::embedding_dense_backward_symint(grad, indices, std::move(num_weights), padding_idx, scale_grad_by_freq);
}
} // namespace acl_op
