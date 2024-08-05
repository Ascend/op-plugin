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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

c10::SmallVector<int64_t, SIZE> get_output_size(const at::Tensor &weight, const at::Tensor &indices,
                                                const at::Tensor &offsets, bool include_last_offset)
{
    TORCH_CHECK(weight.dim() == 2, "weight has to be a 2D Tensor, but got Tensor of dimension ", weight.dim());
    c10::SmallVector<int64_t, SIZE> outputSize = {};
    int64_t offset_size = offsets.size(0);
    if (include_last_offset) {
        offset_size = offsets.size(0) - 1;
    }
    outputSize = {offset_size, weight.size(1)};
    return outputSize;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _embedding_bag(
    const at::Tensor &weight, const at::Tensor &indices, const at::Tensor &offsets, bool scale_grad_by_freq,
    int64_t mode, bool sparse, const c10::optional<at::Tensor> &per_sample_weights, bool include_last_offset,
    int64_t padding_idx)
{
    DO_COMPATIBILITY(aclnnEmbeddingBag, acl_op::_embedding_bag(weight, indices, offsets, scale_grad_by_freq,
    mode, sparse, per_sample_weights, include_last_offset, padding_idx));

    TORCH_CHECK((indices.dim() > 0), "indices.dim() must be greater than 0");
    TORCH_CHECK((weight.dim() > 0), "weight.dim() must be greater than 0");
    TORCH_CHECK((offsets.dim() > 0), "offsets.dim() must be greater than 0");
    c10::SmallVector<int64_t, SIZE> result_size = get_output_size(weight, indices, offsets, include_last_offset);

    at::Tensor output_tensor = npu_preparation::apply_tensor(weight, result_size);
    at::Tensor offset2bag = npu_preparation::apply_tensor(indices, indices.size(0));

    at::Tensor bag_size;
    if (include_last_offset) {
        bag_size = npu_preparation::apply_tensor(offsets, offsets.size(0) - 1);
    } else {
        bag_size = npu_preparation::apply_tensor(offsets);
    }

    at::Tensor max_indices;
    if (mode == 0 || mode == 1) {
        max_indices = npu_preparation::apply_tensor(offsets);
        if (include_last_offset) {
            max_indices = npu_preparation::apply_tensor(offsets, offsets.size(0) - 1);
        }
    } else {
        c10::SmallVector<int64_t, SIZE> max_indices_size =
            get_output_size(weight, indices, offsets, include_last_offset);
        max_indices = npu_preparation::apply_tensor(offsets, max_indices_size);
    }

    at::Tensor offset2bag_cast = offset2bag;
    at::Tensor bag_size_cast = bag_size;
    at::Tensor max_indices_cast = max_indices;

    if (indices.dtype() == at::kLong || offsets.dtype() == at::kLong) {
        offset2bag_cast = offset2bag_cast.to(at::kLong);
        bag_size_cast = bag_size_cast.to(at::kLong);
        max_indices_cast = max_indices_cast.to(at::kLong);
    } else if (indices.dtype() == at::kInt || offsets.dtype() == at::kInt) {
        offset2bag_cast = offset2bag_cast.to(at::kInt);
        bag_size_cast = bag_size_cast.to(at::kInt);
        max_indices_cast = max_indices_cast.to(at::kInt);
    }
    if (mode == 0 && padding_idx < 0) {
        offset2bag_cast = npu_preparation::apply_tensor(offset2bag_cast, 0);
    }

    EXEC_NPU_CMD(aclnnEmbeddingBag, weight, indices, offsets, scale_grad_by_freq,
                 mode, sparse, per_sample_weights, include_last_offset, padding_idx,
                 output_tensor, offset2bag_cast, bag_size_cast, max_indices_cast);
    return std::tie(output_tensor, offset2bag_cast, bag_size_cast, max_indices_cast);
}
}