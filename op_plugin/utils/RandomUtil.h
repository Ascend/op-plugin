// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OP_PLUGIN_UTILS_OPAPI_RANDOM_UTIL_H_
#define OP_PLUGIN_UTILS_OPAPI_RANDOM_UTIL_H_

#include <ATen/Tensor.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>
#include "op_plugin/utils/op_api_common_base.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"

namespace op_plugin {
namespace utils {

static const int64_t BLOCK_SIZE = 256;
static const int64_t MAX_THREADS_PER_MULTI_PROCESSOR = 2048;
static const int64_t MAX_PROCESSOR_COUNT = 78;
static const int64_t UNROLL_2 = 2;
static const int64_t UNROLL_4 = 4;
static const int64_t RAND_OFFSET_PER_CALL = 4;
static const int64_t LEGACY_RAND_COUNTER_OFFSET = 10;
static const int32_t CANN_RANDOM_MODE_LEGACY = 0;
static const int32_t CANN_RANDOM_MODE_COMPAT = 1;
static const int64_t INT32_MAX_VALUE = 2147483647LL;
static const int64_t MAX_DIMS = 8;
static const int64_t RAND_INT64_THRESHOLD = 268435456LL;

struct TensorIterInfo {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int64_t ndim;
    int64_t numel;
    int64_t element_size;
    int unroll;

    TensorIterInfo() : ndim(0), numel(0), element_size(0), unroll(UNROLL_4) {
        for (int i = 0; i < MAX_DIMS; i++) {
            shape[i] = 1;
            strides[i] = 0;
        }
    }

    bool can_use_32bit_indexing() const {
        if (numel > INT32_MAX_VALUE) {
            return false;
        }
        int64_t max_offset = 1;
        for (int64_t i = 0; i < ndim; i++) {
            if (shape[i] > 1) {
                max_offset += (shape[i] - 1) * std::abs(strides[i]) * element_size;
            }
        }
        return max_offset <= INT32_MAX_VALUE;
    }

    int64_t get_dim_to_split() const {
        int64_t max_extent = -1;
        int64_t split_dim = -1;
        for (int64_t dim = ndim - 1; dim >= 0; dim--) {
            if (shape[dim] >= 2) {
                int64_t extent = (shape[dim] - 1) * std::abs(strides[dim]) * element_size;
                if (extent > max_extent) {
                    max_extent = extent;
                    split_dim = dim;
                }
            }
        }
        return split_dim;
    }

    void narrow(int64_t dim, int64_t start, int64_t size) {
        if (dim < 0 || dim >= ndim || size < 1) {
            return;
        }
        numel = numel / shape[dim] * size;
        shape[dim] = size;
    }
};

inline int64_t calc_counter_offset(int64_t nelem, int unroll) {
    unsigned int blocks_per_sm = MAX_THREADS_PER_MULTI_PROCESSOR / BLOCK_SIZE;
    unsigned int grid_x = (nelem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_x = std::min((unsigned int)MAX_PROCESSOR_COUNT * blocks_per_sm, grid_x);
    return ((nelem - 1) / (BLOCK_SIZE * grid_x * unroll) + 1) * RAND_OFFSET_PER_CALL;
}

inline int64_t calc_split_counter_offset(const TensorIterInfo& iter) {
    if (iter.can_use_32bit_indexing()) {
        return calc_counter_offset(iter.numel, iter.unroll);
    }

    TensorIterInfo cur = iter;
    int64_t split_dim = cur.get_dim_to_split();
    if (split_dim < 0) {
        return calc_counter_offset(iter.numel, iter.unroll);
    }

    int64_t left_size = cur.shape[split_dim] / 2;
    int64_t right_size = cur.shape[split_dim] - left_size;

    TensorIterInfo left = cur;
    left.narrow(split_dim, 0, left_size);

    cur.narrow(split_dim, left_size, right_size);

    return calc_split_counter_offset(left) + calc_split_counter_offset(cur);
}

inline bool ConfigureCannRandomnessOnce(bool legacy_impl)
{
    static std::once_flag cann_randomness_compat_flag;
    static bool cann_randomness_compat_configured = false;
    const int32_t cann_random_mode = legacy_impl ? CANN_RANDOM_MODE_LEGACY : CANN_RANDOM_MODE_COMPAT;
    std::call_once(cann_randomness_compat_flag, [cann_random_mode]() {
        using AclnnSetPytorchRandomFunc = int32_t (*)(int32_t);
        static const auto func_addr = GetOpApiFuncAddr("aclnnSetPytorchRandom");
        if (func_addr == nullptr) {
            ASCEND_LOGI("GetOpApiFuncAddr aclnnSetPytorchRandom failed. Randomness CANN random mode config is skipped.");
            return;
        }
        auto aclnn_set_pytorch_random = reinterpret_cast<AclnnSetPytorchRandomFunc>(func_addr);
        NPU_CHECK_ERROR(aclnn_set_pytorch_random(cann_random_mode), "call aclnnSetPytorchRandom failed");
        ASCEND_LOGI("aclnnSetPytorchRandom configured successfully, value=%d.", cann_random_mode);
        cann_randomness_compat_configured = true;
    });
    return cann_randomness_compat_configured;
}

inline bool IsRandomnessLegacyImpl()
{
    static const bool legacy_impl = at_npu::native::env::CheckCompatibleImplBlackListFor(
        at_npu::native::env::CompatibleKey::Randomness);
    return legacy_impl;
}

inline bool PrepareRandomnessCompatible(bool support_cann_random_switch = true)
{
    const bool is_aclnn_only = c10_npu::IsAclnnOnly();
    if (support_cann_random_switch && is_aclnn_only) {
        const bool legacy_impl = IsRandomnessLegacyImpl();
        bool cann_randomness_configured = ConfigureCannRandomnessOnce(legacy_impl);
        if (!cann_randomness_configured) {
            return true;
        }
        return !legacy_impl;
    }
    return is_aclnn_only;
}

inline int64_t calc_final_counter_offset(at::Tensor& self, bool support_cann_random_switch = true,
                                         int64_t from = 0, int64_t to = 0, bool use_from_to = false)
{
    if (!PrepareRandomnessCompatible(support_cann_random_switch)) {
        return LEGACY_RAND_COUNTER_OFFSET;
    }
    TensorIterInfo iter_info;
    iter_info.ndim = self.dim();
    iter_info.numel = self.numel();
    iter_info.element_size = self.itemsize();
    if (iter_info.numel == 0) {
        return 0;
    }
    if (use_from_to) {
        iter_info.unroll = ((to - from) >= RAND_INT64_THRESHOLD) ? UNROLL_2 : UNROLL_4;
    } else {
        iter_info.unroll = (self.scalar_type() == at::kLong) ? UNROLL_2 : UNROLL_4;
    }
    for (int64_t i = 0; i < iter_info.ndim && i < MAX_DIMS; i++) {
        iter_info.shape[i] = self.size(i);
        iter_info.strides[i] = self.stride(i);
    }

    int64_t counter_offset = calc_counter_offset(iter_info.numel, iter_info.unroll);
    if (!iter_info.can_use_32bit_indexing()) {
        counter_offset += calc_split_counter_offset(iter_info);
    }

    return counter_offset;
}

} // namespace utils
} // namespace op_plugin
#endif // OP_PLUGIN_UTILS_OPAPI_RANDOM_UTIL_H_
