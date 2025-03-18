// Copyright (c) 2023-2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void Infer_shape_check(const at::Tensor &y, const at::Tensor &x, const at::Tensor &weight_b,
                       const at::Tensor &indices, const c10::optional<at::Tensor> &weight_a)
{
    TORCH_CHECK(y.dim() == 2 && x.dim() == 2,
        "batch_gather_matmul: Input y and x should be 2D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_b.dim() == 4,
        "batch_gather_matmul: Input weight_b should be 4D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(indices.dim() ==1 && y.size(0) == indices.size(0),
        "batch_gather_matmul: Input indices tensor should be the same size as the y shape 0"
        + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor npu_batch_gather_matmul(
    const at::Tensor& self,
    const at::Tensor& x,
    const at::Tensor& weight_b,
    const at::Tensor& indices,
    const c10::optional<at::Tensor> &weight_a,
    int64_t layer_idx,
    double scale,
    int64_t y_offset,
    int64_t y_slice_size)
{
    Infer_shape_check(self, x, weight_b, indices, weight_a);

    if (y_slice_size == -1) {
        y_slice_size = self.size(1);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);

    EXEC_NPU_CMD(aclnnAddLora, self, x, weight_b, indices, weight_a, layer_idx, scale,  y_offset, y_slice_size, result);
    return self;
}

at::Tensor &npu_batch_gather_matmul_(
    at::Tensor& self,
    const at::Tensor& x,
    const at::Tensor& weight_b,
    const at::Tensor& indices,
    const c10::optional<at::Tensor> &weight_a,
    int64_t layer_idx,
    double scale,
    int64_t y_offset,
    int64_t y_slice_size)
{
    Infer_shape_check(self, x, weight_b, indices, weight_a);

    if (y_slice_size == -1) {
        y_slice_size = self.size(1);
    }

    EXEC_NPU_CMD(aclnnAddLora, self, x, weight_b, indices, weight_a, layer_idx, scale,  y_offset, y_slice_size, self);
    return self;
}
}