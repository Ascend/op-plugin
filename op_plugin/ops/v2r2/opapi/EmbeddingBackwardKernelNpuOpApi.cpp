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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor embedding_backward_symint(const at::Tensor& grad, const at::Tensor& indices, c10::SymInt num_weights,
                                     c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
    DO_COMPATIBILITY(aclnnEmbeddingDenseBackward,
                     acl_op::embedding_backward_symint(grad, indices, num_weights, padding_idx, scale_grad_by_freq,
                                                     sparse));
    TORCH_CHECK(sparse == false, "NPU error, not yet support sparse tensor, when sparse == True" + OPS_ERROR(ErrCode::NOT_SUPPORT));

    int64_t num_weights_int = num_weights.expect_int();
    int64_t padding_idx_int = padding_idx.expect_int();
    // run dense tensor backward
    return op_api::embedding_dense_backward(grad, indices, num_weights_int, padding_idx_int, scale_grad_by_freq);
}
} // namespace op_api
