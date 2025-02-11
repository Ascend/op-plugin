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
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _pack_padded_sequence(const at::Tensor &input, const at::Tensor &lengths,
                                                         bool batch_first)
{
    TORCH_CHECK(input.dim() >= 2, "Input must have two dims.", input.dim(), OPS_ERROR(ErrCode::PARAM));
    // get the size of B and T, the input size is [T, B, *] or [B, T, *]
    auto batchsize = batch_first ? input.size(0) : input.size(1);
    auto timesize = batch_first ? input.size(1) : input.size(0);

    TORCH_CHECK(input.numel() > 0, "Cannot pack empty tensors." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.numel() < std::numeric_limits<int64_t>::max(),
                "Input tensor contain more than the max number of int64." + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(lengths.size(0) == batchsize, "Expected 'len(lengths)' to be equal to batch_size, but got ",
                lengths.size(0), " (batch_size=", batchsize, ")" + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(lengths.device().type() == at::kCPU,
                "'lengths' argument should be a CPU tensor, but got ",
                lengths.device().str(), " tensor" + OPS_ERROR(ErrCode::PARAM));
    auto lengths_vec = lengths.contiguous().data_ptr<int64_t>();
    TORCH_CHECK(lengths_vec != nullptr && lengths_vec[batchsize - 1] > 0,
                "Length of all samples has to be greater than 0, but found an element "
                "in 'lengths' that is <= 0" + OPS_ERROR(ErrCode::PARAM));

    // According to the TMG decision, adaptation avoidance scheme I
    // is temporarily adopted to retain the filling within effective T0,
    // [B*T0, *]
    auto output = batch_first ? input.transpose(0, 1) : input;
    auto len = lengths_vec[0];
    if (len < timesize) {
        vector<int> tmp_vector = {};
        for (int i = 0; i < len; i++) {
            tmp_vector.emplace_back(i);
        }
        auto index = at::from_blob(tmp_vector.data(), {len}, at::kInt);
        index = npu_preparation::copy_tensor_host_to_device(index);
        output = op_plugin::index_select(output, 0, index);
        timesize = len;
    }

    at::SmallVector<int64_t, N> shape;
    shape.emplace_back(batchsize * timesize);
    for (int i = 2; i < input.dim(); i++) {
        shape.emplace_back(input.size(i));
    }

    output = output.contiguous();
    output = output.reshape(shape);

    at::Tensor batchsizes = at::empty({timesize}, lengths.options());
    auto batchsize_vec = batchsizes.data_ptr<int64_t>();
    TORCH_CHECK(batchsize_vec != nullptr, "batchsizes is null" + OPS_ERROR(ErrCode::PARAM));
    int64_t last = batchsize - 1;
    for (int ti = 0; ti < timesize; ti++) {
        for (int bi = last; bi >= 0; bi--) {
            if (lengths_vec[bi] > ti) {
                batchsize_vec[ti] = (bi + 1);
                last = bi;
                break;
            }
        }
    }
    return std::tie(output, batchsizes);
}
} // namespace acl_op
