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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor embedding_dense_backward(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq)
{
    TORCH_CHECK(grad_output.dim() >= 1, "The dim of input 'grad_output' must be greater than or equal to 1."
        + OPS_ERROR(ErrCode::PARAM));
    auto output_size = {num_weights, grad_output.size(-1)};
    at::Tensor result = npu_preparation::apply_tensor(grad_output, output_size);

    // indices must be int64 in pytorch, but npu can only support int32
    auto indices_int32 = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kInt);
    at_npu::native::OpCommand cmd;
    cmd.Name("EmbeddingDenseGrad")
        .Input(grad_output)
        .Input(indices_int32)
        .Attr("num_weights", num_weights)
        .Attr("padding_idx", padding_idx)
        .Attr("scale_grad_by_freq", scale_grad_by_freq)
        .Output(result)
        .Run();
    return result;
}
} // namespace acl_op
