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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t INT4_NUMS_IN_INT32 = 8;
const int64_t WEIGHT_SHAPE_SIZE = 2;
const int64_t CUBE_BLOCK_SIZE = 16;
const int64_t C0_SIZE_INT32 = 8;

void convert_to_int4_pack(const std::vector<int32_t>& weight_data, std::vector<int32_t>& weight_int4pack_data)
{
    size_t n = weight_int4pack_data.size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t a = static_cast<uint32_t>(weight_data[i * 8]);
        uint32_t b = static_cast<uint32_t>(weight_data[i * 8 + 1]);
        uint32_t c = static_cast<uint32_t>(weight_data[i * 8 + 2]);
        uint32_t d = static_cast<uint32_t>(weight_data[i * 8 + 3]);
        uint32_t e = static_cast<uint32_t>(weight_data[i * 8 + 4]);
        uint32_t f = static_cast<uint32_t>(weight_data[i * 8 + 5]);
        uint32_t g = static_cast<uint32_t>(weight_data[i * 8 + 6]);
        uint32_t h = static_cast<uint32_t>(weight_data[i * 8 + 7]);

        weight_int4pack_data[i] = (a & 0xF) | (b & 0xF) << 4 | (c & 0xF) << 8 | (d & 0xF) << 12 |
                                (e & 0xF) << 16 | (f & 0xF) << 20 | (g & 0xF) << 24 | (h & 0xF) << 28;
    }
}

void trans_nd_to_nz(std::vector<int32_t>& weight_array, uint64_t k, uint64_t n)
{
    uint64_t k1 = (k + CUBE_BLOCK_SIZE - 1) / CUBE_BLOCK_SIZE;
    int64_t weight_nz_size = op_infer::CeilDiv(k, CUBE_BLOCK_SIZE) *
                           op_infer::CeilDiv(n, C0_SIZE_INT32) * CUBE_BLOCK_SIZE * C0_SIZE_INT32;
    std::vector<int32_t> weight_nz_array(weight_nz_size, 0);

    // (k, n) -> (n1, k1, k0, n0)
    // (k, n) -> (ceil_div(n, 8), ceil_div(k, 16), 16, 8)
    for (size_t idx = 0; idx < weight_array.size(); ++idx) {
        size_t idx_k = idx / n;
        size_t idx_n = idx % n;
        size_t idx_k0 = idx_k % CUBE_BLOCK_SIZE;
        size_t idx_k1 = idx_k / CUBE_BLOCK_SIZE;
        size_t idx_n0 = idx_n % C0_SIZE_INT32;
        size_t idx_n1 = idx_n / C0_SIZE_INT32;
        weight_nz_array[idx_n1 * k1 * CUBE_BLOCK_SIZE * C0_SIZE_INT32 + idx_k1 * CUBE_BLOCK_SIZE * C0_SIZE_INT32 +
            idx_k0 * C0_SIZE_INT32 + idx_n0] = weight_array[idx];
    }
    weight_array = weight_nz_array;
}

inline void int4pack_params_check(const at::Tensor &weight)
{
    TORCH_CHECK(weight.is_contiguous(), "weight should be contiguous", OPS_ERROR(ErrCode::PARAM));
    auto weight_dim_num = weight.dim();
    TORCH_CHECK(weight_dim_num == WEIGHT_SHAPE_SIZE, "weight shape only support dim num 2, but it is ",
                weight_dim_num, OPS_ERROR(ErrCode::PARAM));

    auto weight_dtype = weight.dtype();
    TORCH_CHECK(weight_dtype == at::kInt, "weight dtype only support int32, but it is ", weight_dtype,
                OPS_ERROR(ErrCode::TYPE));

    auto weight_first_dim = weight.size(weight_dim_num - 2);
    auto weight_last_dim = weight.size(weight_dim_num - 1);
    TORCH_CHECK(weight_first_dim > 0 && weight_last_dim > 0, "weight dim should be greater than 0",
                OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(weight_last_dim % INT4_NUMS_IN_INT32 == 0, "weight last dim should be the multiple of 8, but it is ",
                weight_last_dim, OPS_ERROR(ErrCode::PARAM));
}

at::Tensor npu_convert_weight_to_int4pack(const at::Tensor &weight, int64_t inner_k_tiles)
{
    int4pack_params_check(weight);
    auto weight_dim_num = weight.dim();
    auto weight_first_dim = weight.size(weight_dim_num - 2);
    auto weight_last_dim = weight.size(weight_dim_num - 1);
    int64_t weight_format = at_npu::native::custom_ops::get_npu_format(weight);
    at::Tensor weight_nd;
    bool is_weight_nz = (weight_format == ACL_FORMAT_FRACTAL_NZ);
    if (is_weight_nz) {
        weight_nd = at_npu::native::custom_ops::npu_format_cast(weight, ACL_FORMAT_ND);
    }

    // trans nd to nz on cpu, because the c0 size of int32(int4pack) fractal_nz is 8, which is not support yet on npu
    at::Tensor weight_cpu = is_weight_nz ? weight_nd.to("cpu") : weight.to("cpu");
    std::vector<int32_t> weight_data(weight_cpu.data_ptr<int32_t>(), weight_cpu.data_ptr<int32_t>() + weight_cpu.numel());
    // store 8 int4 numbers in sequence into an int32
    std::vector<int32_t> weight_int4pack_data(weight_data.size() / INT4_NUMS_IN_INT32, 0);
    std::vector<int64_t> weight_int4pack_shape = {weight_first_dim, weight_last_dim / INT4_NUMS_IN_INT32};

    convert_to_int4_pack(weight_data, weight_int4pack_data);

    if (is_weight_nz) {
        trans_nd_to_nz(weight_int4pack_data, weight_first_dim, weight_last_dim / INT4_NUMS_IN_INT32);
        weight_int4pack_shape = {op_infer::CeilDiv(weight_last_dim / INT4_NUMS_IN_INT32, C0_SIZE_INT32),
                               op_infer::CeilDiv(weight_first_dim, CUBE_BLOCK_SIZE), CUBE_BLOCK_SIZE, C0_SIZE_INT32};
    }

    c10::TensorOptions options_cpu = weight_cpu.options().dtype(at::kInt);
    at::Tensor weight_int4_pack_cpu = at::from_blob(weight_int4pack_data.data(), weight_int4pack_shape,
        options_cpu);
    auto output_size = op_infer::array_to_small_vector(weight_int4pack_shape);
    c10::TensorOptions options = weight.options().dtype(at::kInt);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    if (is_weight_nz) {
        // datacopy from cpu to npu with aclrtMemcpy. copy_ will cause insertion of wrong transdata with c0=16.
        int64_t nbytes = result.numel() * result.element_size();
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
        NPU_CHECK_ERROR(aclrtMemcpy(const_cast<void*>(result.storage().unsafeGetStorageImpl()->data()), nbytes,
            weight_int4_pack_cpu.storage().unsafeGetStorageImpl()->data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));
        // set storage format, stride, shape
        auto &out_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(result);
        out_desc.npu_format_ = ACL_FORMAT_FRACTAL_NZ;
        out_desc.origin_format_ = ACL_FORMAT_ND;
        out_desc.base_sizes_ = {weight_first_dim, weight_last_dim / INT4_NUMS_IN_INT32};
        out_desc.base_strides_ = {weight_last_dim / INT4_NUMS_IN_INT32, 1};
        result.set_(result.storage(), 0, {weight_first_dim, weight_last_dim / INT4_NUMS_IN_INT32},
            {weight_last_dim / INT4_NUMS_IN_INT32, 1});
    } else {
        result.copy_(weight_int4_pack_cpu);
    }
    return result;
}
}  // namespace op_api
