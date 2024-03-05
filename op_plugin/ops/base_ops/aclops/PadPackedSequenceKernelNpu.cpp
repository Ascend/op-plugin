// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
std::tuple<at::Tensor, at::Tensor> _pad_packed_sequence(const at::Tensor &input, const at::Tensor &batch_sizes,
                                                        bool batch_first, const at::Scalar &padding_value,
                                                        int64_t total_length)
{
    TORCH_CHECK(batch_sizes.dim() >= 1, "Input batch_sizes dim number must larger than one."
    + OPS_ERROR(ErrCode::PARAM));
    if (total_length > 0) {
        TORCH_CHECK(total_length >= batch_sizes.size(0),
            "Expected total_length to be at least the length of the longest "
            "sequence in input, but got total_length=",
            total_length,
            " and "
            "max sequence length being ",
            batch_sizes.size(0), OPS_ERROR(ErrCode::PARAM));
    }

    // input shape is [B*T, *], calculate the B and T
    auto batch_sizes_cpu = batch_sizes.to("cpu");
    int64_t *batch_size_vec = batch_sizes_cpu.data_ptr<int64_t>();
    TORCH_CHECK(batch_size_vec != nullptr, "batch_sizes is null" + OPS_ERROR(ErrCode::PARAM));
    auto batchsize = batch_size_vec[0];
    auto timesize = batch_sizes.size(0);

    // make tensor after padding, [B, T, *] or [T, B, *]
    at::SmallVector<int64_t, N> shape;
    shape.emplace_back(timesize);
    shape.emplace_back(batchsize);

    for (int i = 1; i < input.dim(); i++) {
        shape.emplace_back(input.size(i));
    }

    auto output = input.reshape(shape);
    if (batch_first) {
        output = output.transpose(0, 1);
    }
    output = output.contiguous();

    auto batch_sizes_val = at::empty({batchsize}, batch_sizes_cpu.options());
    auto batch_sizes_vec = batch_sizes_val.data_ptr<int64_t>();
    TORCH_CHECK(batch_sizes_vec != nullptr, "batch_sizes_val is null" + OPS_ERROR(ErrCode::PARAM));
    int64_t last = timesize - 1;
    for (int bi = 0; bi < batchsize; bi++) {
        for (int ti = last; ti >= 0; ti--) {
            if (batch_size_vec[ti] > bi) {
                batch_sizes_vec[bi] = (ti + 1);
                last = ti;
                break;
            }
        }
    }
    return std::tie(output, batch_sizes_val);
}
} // namespace acl_op
