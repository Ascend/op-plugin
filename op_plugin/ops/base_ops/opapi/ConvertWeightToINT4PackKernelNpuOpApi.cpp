// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t INT4_NUMS_IN_INT32 = 8;
const int64_t WEIGHT_SHAPE_SIZE = 2;
void convertToINT4Pack(const std::vector<int32_t>& weightData, std::vector<int32_t>& weightInt4PackData)
{
    size_t n = weightInt4PackData.size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t  a = static_cast<uint32_t>(weightData[i * 8]);
        uint32_t  b = static_cast<uint32_t>(weightData[i * 8 + 1]);
        uint32_t  c = static_cast<uint32_t>(weightData[i * 8 + 2]);
        uint32_t  d = static_cast<uint32_t>(weightData[i * 8 + 3]);
        uint32_t  e = static_cast<uint32_t>(weightData[i * 8 + 4]);
        uint32_t  f = static_cast<uint32_t>(weightData[i * 8 + 5]);
        uint32_t  g = static_cast<uint32_t>(weightData[i * 8 + 6]);
        uint32_t  h = static_cast<uint32_t>(weightData[i * 8 + 7]);

        weightInt4PackData[i] = (a & 0xF) | (b & 0xF) << 4 | (c & 0xF) << 8 | (d & 0xF) << 12 |
                                (e & 0xF) << 16 | (f & 0xF) << 20 | (g & 0xF) << 24 | (h & 0xF) << 28;
    }
}

at::Tensor npu_convert_weight_to_int4pack(const at::Tensor &weight, int64_t inner_k_tiles)
{
    auto weight_dim_num = weight.dim();
    auto weight_dtype = weight.dtype();
    auto weight_last_dim = weight.size(weight_dim_num - 1);
    TORCH_CHECK(weight_dim_num == WEIGHT_SHAPE_SIZE, "weight shape only support dim num 2, but it is ",
                weight_dim_num, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_dtype == at::kInt, "weight dtype only support int32, but it is ", weight_dtype,
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(weight_last_dim % INT4_NUMS_IN_INT32 == 0, "weight last dim should be the multiple of 8, but it is ",
                weight_last_dim, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.is_contiguous(), "weight should be contiguous", OPS_ERROR(ErrCode::PARAM));

    at::Tensor weight_cpu = weight.to("cpu");
    std::vector<int32_t> weightData(weight_cpu.data_ptr<int32_t>(), weight_cpu.data_ptr<int32_t>() + weight_cpu.numel());
    // store 8 int4 numbers in sequence into an int32
    std::vector<int32_t> weightInt4PackData(weightData.size() / INT4_NUMS_IN_INT32, 0);

    convertToINT4Pack(weightData, weightInt4PackData);
    c10::TensorOptions options_cpu = weight_cpu.options().dtype(at::kInt);
    at::Tensor weight_int4_pack_cpu = at::from_blob(weightInt4PackData.data(),
        {weight.size(0), weight.size(1) / INT4_NUMS_IN_INT32}, options_cpu).clone();
    auto output_size = op_infer::array_to_small_vector({weight.size(0), weight.size(1) / INT4_NUMS_IN_INT32});
    c10::TensorOptions options = weight.options().dtype(at::kInt);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);
    result.copy_(weight_int4_pack_cpu);
    return result;
}
}  // namespace op_api