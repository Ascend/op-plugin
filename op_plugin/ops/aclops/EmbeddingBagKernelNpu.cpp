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

namespace {

c10::SmallVector<int64_t, SIZE> _embedding_bag_npu_output_size(const at::Tensor &weight, const at::Tensor &indices,
                                                               const at::Tensor &offsets, bool include_last_offset)
{
    TORCH_CHECK(weight.dim() == 2, "weight has to be a 2D Tensor, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> outputSize = {};
    if (indices.dim() == 1) {
        int64_t offset_size = offsets.size(0);
        if (include_last_offset) {
            offset_size = offsets.size(0) - 1;
        }
        outputSize = {offset_size, weight.size(1)};
    } else {
        outputSize = {indices.size(0), weight.size(1)};
    }
    return outputSize;
}

string get_mode_str(const int64_t mode)
{
    string modeStr = "mean";
    if (mode == 0) {
        modeStr = "sum";
    } else if (mode == 1) {
        modeStr = "mean";
    } else {
        modeStr = "max";
    }
    return modeStr;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _embedding_bag_out_npu_nocheck(
    const at::Tensor &weight, const at::Tensor &indices, const at::Tensor &offsets, bool scale_grad_by_freq,
    int64_t mode, bool sparse, const at::Tensor &per_sample_weights, bool include_last_offset, int64_t padding_idx,
    at::Tensor &output, at::Tensor &offset2bag, at::Tensor &bag_size, at::Tensor &max_indices,
    const at::Tensor &indices_ori, const at::Tensor &offsets_ori)
{
    string mode_str = get_mode_str(mode);
    at_npu::native::OpCommand cmd;

    if (indices.numel() == 0 || offsets.numel() == 0) {
        TORCH_CHECK(mode == 0, "The mode must be sum" + OPS_ERROR(ErrCode::PARAM));
        output = npu_preparation::apply_tensor(weight);
        acl_op::fill_(output, 0);
        offset2bag = npu_preparation::apply_tensor(indices, 0);
        bag_size = npu_preparation::apply_tensor(indices, 0);
        max_indices = npu_preparation::apply_tensor(indices, 0);
        return std::tie(output, offset2bag, bag_size, max_indices);
    } else {
        cmd.Name("EmbeddingBag").Input(weight).Input(indices).Input(offsets);
        if (per_sample_weights.defined()) {
            cmd.Input(per_sample_weights);
        }
        cmd.Output(output)
            .Output(offset2bag)
            .Output(bag_size)
            .Output(max_indices)
            .Attr("mode", mode_str)
            .Attr("scale_grad_by_freq", scale_grad_by_freq)
            .Attr("sparse", sparse)
            .Attr("include_last_offset", include_last_offset)
            .Attr("padding_idx", padding_idx)
            .Run();
    }

    if (mode_str == "sum" && padding_idx == -1) {
        offset2bag = npu_preparation::apply_tensor(indices, 0);
    }
    at::Tensor offset2bag_cast = const_cast<at::Tensor &>(offset2bag);
    at::Tensor bag_size_cast = const_cast<at::Tensor &>(bag_size);
    at::Tensor max_indices_cast = const_cast<at::Tensor &>(max_indices);
    if (indices_ori.dtype() == at::kLong || offsets_ori.dtype() == at::kLong) {
        offset2bag_cast = offset2bag_cast.to(at::kLong);
        bag_size_cast = bag_size_cast.to(at::kLong);
        max_indices_cast = max_indices_cast.to(at::kLong);
    }

    return std::tie(output, offset2bag_cast, bag_size_cast, max_indices_cast);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _embedding_bag(
    const at::Tensor &weight, const at::Tensor &indices, const at::Tensor &offsets, bool scale_grad_by_freq,
    int64_t mode, bool sparse, const c10::optional<at::Tensor> &per_sample_weights, bool include_last_offset,
    int64_t padding_idx)
{
    TORCH_CHECK((indices.dim() > 0), "indices.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((weight.dim() > 0), "weight.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((offsets.dim() > 0), "offsets.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));

    at::Tensor indices_cast = const_cast<at::Tensor &>(indices);
    at::Tensor offsets_cast = const_cast<at::Tensor &>(offsets);
    if (indices.dtype() == at::kLong) {
        indices_cast = at_npu::native::custom_ops::npu_dtype_cast(indices_cast, at::kInt);
    }
    if (offsets.dtype() == at::kLong) {
        offsets_cast = at_npu::native::custom_ops::npu_dtype_cast(offsets_cast, at::kInt);
    }

    const at::Tensor &per_sample_weights_core = c10::value_or_else(per_sample_weights, [] { return at::Tensor(); });

    c10::SmallVector<int64_t, SIZE> result_size =
        _embedding_bag_npu_output_size(weight, indices_cast, offsets_cast, include_last_offset);

    at::Tensor output_tensor = npu_preparation::apply_tensor(weight, result_size);
    // 申请offset2bag的Tensor
    int64_t indices_num = indices_cast.size(0);

    string mode_str = get_mode_str(mode);

    at::Tensor offset2bag = npu_preparation::apply_tensor(indices_cast);
    at::Tensor bag_size = npu_preparation::apply_tensor(offsets_cast);

    if (include_last_offset) {
        bag_size = npu_preparation::apply_tensor(offsets_cast, offsets_cast.size(0) - 1);
    }

    at::Tensor max_indices;
    if (mode_str == "max") {
        c10::SmallVector<int64_t, SIZE> max_indices_size =
            _embedding_bag_npu_output_size(weight, indices_cast, offsets_cast, include_last_offset);
        max_indices = npu_preparation::apply_tensor(offsets_cast, max_indices_size);
    } else {
        max_indices = npu_preparation::apply_tensor(offsets_cast);
        if (include_last_offset) {
            max_indices = npu_preparation::apply_tensor(offsets_cast, offsets_cast.size(0) - 1);
        }
    }

    return _embedding_bag_out_npu_nocheck(weight, indices_cast, offsets_cast, scale_grad_by_freq, mode, sparse,
                                          per_sample_weights_core, include_last_offset, padding_idx, output_tensor,
                                          offset2bag, bag_size, max_indices, indices, offsets);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> _embedding_bag_forward_only(
    const at::Tensor &weight, const at::Tensor &indices, const at::Tensor &offsets, bool scale_grad_by_freq,
    int64_t mode, bool sparse, const c10::optional<at::Tensor> &per_sample_weights, bool include_last_offset,
    int64_t padding_idx)
{
    TORCH_CHECK((indices.dim() > 0), "indices.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((weight.dim() > 0), "weight.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((offsets.dim() > 0), "offsets.dim() must be greater than 0" + OPS_ERROR(ErrCode::PARAM));

    at::Tensor indices_cast = const_cast<at::Tensor &>(indices);
    at::Tensor offsets_cast = const_cast<at::Tensor &>(offsets);
    if (indices.dtype() == at::kLong) {
        indices_cast = at_npu::native::custom_ops::npu_dtype_cast(indices_cast, at::kInt);
    }
    if (offsets.dtype() == at::kLong) {
        offsets_cast = at_npu::native::custom_ops::npu_dtype_cast(offsets_cast, at::kInt);
    }

    const at::Tensor &per_sample_weights_core = c10::value_or_else(per_sample_weights, [] { return at::Tensor(); });

    c10::SmallVector<int64_t, SIZE> result_size =
        _embedding_bag_npu_output_size(weight, indices_cast, offsets_cast, include_last_offset);

    at::Tensor output_tensor = npu_preparation::apply_tensor(weight, result_size);
    // 申请offset2bag的Tensor
    int64_t indices_num = indices_cast.size(0);

    string mode_str = get_mode_str(mode);

    at::Tensor offset2bag = npu_preparation::apply_tensor(indices_cast);
    at::Tensor bag_size = npu_preparation::apply_tensor(offsets_cast);

    if (include_last_offset) {
        bag_size = npu_preparation::apply_tensor(offsets_cast, offsets_cast.size(0) - 1);
    }

    at::Tensor max_indices;
    if (mode_str == "max") {
        c10::SmallVector<int64_t, SIZE> max_indices_size =
            _embedding_bag_npu_output_size(weight, indices_cast, offsets_cast, include_last_offset);
        max_indices = npu_preparation::apply_tensor(offsets_cast, max_indices_size);
    } else {
        max_indices = npu_preparation::apply_tensor(offsets_cast);
        if (include_last_offset) {
            max_indices = npu_preparation::apply_tensor(offsets_cast, offsets_cast.size(0) - 1);
        }
    }

    return _embedding_bag_out_npu_nocheck(weight, indices_cast, offsets_cast, scale_grad_by_freq, mode, sparse,
                                          per_sample_weights_core, include_last_offset, padding_idx, output_tensor,
                                          offset2bag, bag_size, max_indices, indices, offsets);
}
} // namespace acl_op
