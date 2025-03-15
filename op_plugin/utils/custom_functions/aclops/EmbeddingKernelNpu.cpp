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

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor embedding_common_nocheck(
    const at::Tensor& weight,
    const at::Tensor& indices)
{
    auto output_size = op_infer::array_to_small_vector(indices.sizes());
    TORCH_CHECK(weight.numel() > 0, "The input tensor is an empty tensor.", OPS_ERROR(ErrCode::PARAM));
    output_size.emplace_back(weight.size(weight.dim() - 1));
    at::Tensor result = npu_preparation::apply_tensor(weight, output_size);

    c10::SmallVector<int64_t, N> dim_vec = {0};
    int64_t batch_dims = 0;

    at_npu::native::OpCommand cmd;
    cmd.Name("GatherV2")
        .Input(weight)
        .Input(indices)
        .Input(dim_vec)
        .Output(result)
        .Attr("batch_dims", batch_dims)
        .Run();
    return result;
}
} // namespace acl_op
