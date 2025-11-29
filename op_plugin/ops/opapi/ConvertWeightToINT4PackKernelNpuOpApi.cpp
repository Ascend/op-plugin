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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t INT4_NUMS_IN_INT32 = 8;
const int64_t WEIGHT_SHAPE_SIZE = 2;
const int64_t WEIGHT_SHAPE_SIZE_THREE = 3;
const int64_t CUBE_BLOCK_SIZE = 16;
const int64_t C0_SIZE_INT32 = 8;
const uint32_t FP32_TO_FP4_MASK = 0xFFC00000;
const std::unordered_map<uint32_t, uint32_t> FP32_BIT_TO_FP4_E2M1 = {
    {0x00000000, 0b0000}, // 0.0
    {0x3F000000, 0b0001}, // 0.5
    {0x3F800000, 0b0010}, // 1.0
    {0x3FC00000, 0b0011}, // 1.5
    {0x40000000, 0b0100}, // 2.0
    {0x40400000, 0b0101}, // 3.0
    {0x40800000, 0b0110}, // 4.0
    {0x40C00000, 0b0111}, // 6.0

    {0x80000000, 0b1000}, // -0.0
    {0xBF000000, 0b1001}, // -0.5
    {0xBF800000, 0b1010}, // -1.0
    {0xBFC00000, 0b1011}, // -1.5
    {0xC0000000, 0b1100}, // -2.0
    {0xC0400000, 0b1101}, // -3.0
    {0xC0800000, 0b1110}, // -4.0
    {0xC0C00000, 0b1111}, // -6.0
};

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

uint32_t convert_fp32_to_fp4_e2m1(int32_t data)
{
    uint32_t fp32_bits = data & FP32_TO_FP4_MASK;
    auto it = FP32_BIT_TO_FP4_E2M1.find(fp32_bits);
    if (it != FP32_BIT_TO_FP4_E2M1.end()) {
        return it->second;
    }
    return 0b0000;
}

void convert_to_fp4_pack(const std::vector<int32_t> &weight_data, std::vector<int32_t> &weight_fp4pack_data)
{
    size_t n = weight_fp4pack_data.size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t num1 = convert_fp32_to_fp4_e2m1(weight_data[i * 8]);
        uint32_t num2 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 1]);
        uint32_t num3 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 2]);
        uint32_t num4 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 3]);
        uint32_t num5 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 4]);
        uint32_t num6 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 5]);
        uint32_t num7 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 6]);
        uint32_t num8 = convert_fp32_to_fp4_e2m1(weight_data[i * 8 + 7]);
        // 取8个数的低4位，然后分别在int32的第0，4，8，12，16，20，24，28位开始放
        weight_fp4pack_data[i] = (num1 & 0xF) | (num2 & 0xF) << 4 | (num3 & 0xF) << 8 | (num4 & 0xF) << 12 | (num5 & 0xF) << 16 | (num6 & 0xF) << 20 | (num7 & 0xF) << 24 | (num8 & 0xF) << 28;
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
    TORCH_CHECK(weight_dim_num == WEIGHT_SHAPE_SIZE || weight_dim_num == WEIGHT_SHAPE_SIZE_THREE,
                "weight shape only support dim num 2/3, but it is ", weight_dim_num, OPS_ERROR(ErrCode::PARAM));

    auto weight_dtype = weight.dtype();
    TORCH_CHECK(weight_dtype == at::kInt || weight_dtype == at::kFloat,
        "weight dtype only support int32 and float32, but it is ", weight_dtype, OPS_ERROR(ErrCode::TYPE));

    for (auto idx = 0; idx < weight_dim_num; ++idx) {
        TORCH_CHECK(weight.size(idx) > 0, "weight dim should be greater than 0", OPS_ERROR(ErrCode::PARAM));
        if (idx == weight_dim_num - 1) {
            TORCH_CHECK(weight.size(idx) % INT4_NUMS_IN_INT32 == 0,
                        "weight last dim should be the multiple of 8, but it is ", weight.size(idx),
                        OPS_ERROR(ErrCode::PARAM));
        }
    }
}

int64_t get_element_size(const at::Tensor &tensor)
{
    int64_t shape_size = 1;
    for (auto idx = 0; idx < tensor.dim(); ++idx) {
        shape_size *= tensor.size(idx);
    }
    return shape_size;
}

at::Tensor npu_convert_weight_to_b4pack(const at::Tensor &weight)
{
    // 1）int32（int4）ND/NZ
    // 2）float32（float4_e2m1）ND/NZ

    auto weight_dim_num = weight.dim();
    int64_t weight_format = at_npu::native::custom_ops::get_npu_format(weight);
    bool weight_nz_flag = (weight_format == ACL_FORMAT_FRACTAL_NZ) ||
                          (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_16) ||
                          (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_32);
    bool supported_format =  weight_nz_flag || weight_format == ACL_FORMAT_ND || weight_format == ACL_FORMAT_NCL;
    TORCH_CHECK(supported_format,
        "weight_format only support ND/NCL/NZ/NZ_C0_16/NZ_C0_32, but it is ", weight_format, OPS_ERROR(ErrCode::PARAM));

    int64_t weight_elem_size = get_element_size(weight);
    int64_t weight_bytes = weight_elem_size * sizeof(int32_t);

    // device to host
    at::Tensor weight_cpu;
    std::vector<int32_t> weight_data(weight_elem_size, 0);
    if (weight_nz_flag) {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
        TORCH_CHECK(weight.storage().unsafeGetStorageImpl() != nullptr, "Failed to get weight storage pointer",
            OPS_ERROR(ErrCode::PARAM));
        NPU_CHECK_ERROR(aclrtMemcpy(weight_data.data(), weight_bytes, weight.storage().unsafeGetStorageImpl()->data(),
            weight_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        weight_cpu = weight.to("cpu");
        if (weight.dtype() == at::kInt) {
            weight_data = std::vector<int32_t>(
                weight_cpu.data_ptr<int32_t>(), weight_cpu.data_ptr<int32_t>() + weight_cpu.numel());
        } else {
            std::vector<float> weight_data_f32(
                weight_cpu.data_ptr<float>(), weight_cpu.data_ptr<float>() + weight_cpu.numel());
            weight_data.resize(weight_data_f32.size(), 0);
            std::memcpy(weight_data.data(), weight_data_f32.data(), weight_data_f32.size() * sizeof(float));
        }
    }

    // pack
    std::vector<int32_t> packed_weight(weight_data.size() / INT4_NUMS_IN_INT32, 0);
    if (weight.dtype() == at::kInt) {
        // store 8 int4 numbers in sequence into an int32
        convert_to_int4_pack(weight_data, packed_weight);
    } else {
        // store 8 fp4_e2m1 numbers in sequence into an float32
        convert_to_fp4_pack(weight_data, packed_weight);
    }

    // update packed shape, last dim is divided by 8
    std::vector<int64_t> weight_before_packed_shape;
    std::vector<int64_t> weight_packed_shape;
    if (weight_nz_flag) {
        auto storage_shape = torch_npu::NPUBridge::GetNpuStorageImpl(weight)->npu_desc_.storage_sizes_;
        auto dim_num = storage_shape.size();
        TORCH_CHECK(dim_num > 1, "nz storage shape dim should be greater than 1", OPS_ERROR(ErrCode::PARAM));
        for (auto data : storage_shape) {
            weight_before_packed_shape.push_back(data);
            weight_packed_shape.push_back(data);
        }
        weight_packed_shape[dim_num - 1] = op_infer::CeilDiv(weight_packed_shape[dim_num - 1], INT4_NUMS_IN_INT32);
    } else {
        for (auto idx = 0; idx < weight_dim_num; ++idx) {
            weight_before_packed_shape.push_back(weight.size(idx));
            weight_packed_shape.push_back(weight.size(idx));
        }
        weight_packed_shape[weight_dim_num - 1] /= INT4_NUMS_IN_INT32;
    }
    ASCEND_LOGI("before pack storage shape: %s", op_plugin::utils::get_vector_str(weight_before_packed_shape).c_str());
    ASCEND_LOGI("after pack storage shape: %s", op_plugin::utils::get_vector_str(weight_packed_shape).c_str());

    // host to device
    auto weight_packed_vec = op_infer::array_to_small_vector(weight_packed_shape);
    c10::TensorOptions weight_packed_option = weight.options().dtype(weight.dtype());
    at::Tensor weight_packed_npu =
        npu_preparation::apply_tensor_without_format(weight_packed_vec, weight_packed_option);
    if (weight_nz_flag) {
        // copy_ does not support internal format, use aclrtMemcpy instead
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
        TORCH_CHECK(weight_packed_npu.storage().unsafeGetStorageImpl() != nullptr,
            "Failed to get weight_packed_npu storage pointer", OPS_ERROR(ErrCode::PARAM));
        int64_t weight_packed_bytes = get_element_size(weight_packed_npu) * sizeof(int32_t);
        NPU_CHECK_ERROR(aclrtMemcpy(const_cast<void *>(weight_packed_npu.storage().unsafeGetStorageImpl()->data()),
            weight_packed_bytes, packed_weight.data(), weight_packed_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

        auto &weight_packed_npu_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(weight_packed_npu);
        auto npu_format = ACL_FORMAT_FRACTAL_NZ;
        if (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_32) {
            npu_format = ACL_FORMAT_FRACTAL_NZ_C0_4;
        } else if (weight_format == ACL_FORMAT_FRACTAL_NZ_C0_16) {
            npu_format = ACL_FORMAT_FRACTAL_NZ_C0_2;
        }

        weight_packed_npu_desc.npu_format_ = npu_format;
        weight_packed_npu_desc.origin_format_ = ACL_FORMAT_ND;
        std::vector<int64_t> weight_packed_npu_sizes(weight_dim_num);
        std::vector<int64_t> weight_packed_npu_strides(weight_dim_num);
        int64_t stride = 1;
        for (auto idx = weight_dim_num - 1; idx >= 0; --idx) {
            int64_t dim_value =
                (idx == weight_dim_num - 1) ? (weight.size(idx) / INT4_NUMS_IN_INT32) : weight.size(idx);
            weight_packed_npu_sizes[idx] = dim_value;
            weight_packed_npu_strides[idx] = stride;
            stride *= dim_value;
        }
        weight_packed_npu_desc.base_sizes_ = weight_packed_npu_sizes;
        weight_packed_npu_desc.base_strides_ = weight_packed_npu_strides;
        weight_packed_npu.set_(weight_packed_npu.storage(), 0, weight_packed_npu_sizes, weight_packed_npu_strides);
    } else {
        c10::TensorOptions options_cpu = weight_cpu.options().dtype(weight.dtype());
        at::Tensor weight_packed_cpu = at::from_blob(packed_weight.data(), weight_packed_shape, options_cpu);
        weight_packed_npu.copy_(weight_packed_cpu);
    }

    return weight_packed_npu;
}

at::Tensor npu_convert_weight_to_int4pack(const at::Tensor &weight, int64_t inner_k_tiles)
{
    int4pack_params_check(weight);
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_95) {
        return npu_convert_weight_to_b4pack(weight);
    }
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
    std::vector<int32_t> weight_data(
        weight_cpu.data_ptr<int32_t>(), weight_cpu.data_ptr<int32_t>() + weight_cpu.numel());
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
