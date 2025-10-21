// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_scatter_pa_kv_cache_functional(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const c10::optional<at::Tensor>& compress_lens,
    const c10::optional<at::Tensor>& compress_seq_offsets,
    const c10::optional<at::Tensor>& seq_lens)
{
    char* cache_mode = "PA_NZ";
    char* scatter_mode = "None";
    c10::SmallVector<int64_t, op_infer::SIZE> strides_size = {1, 1};
    at::IntArrayRef strides = at::IntArrayRef(strides_size);
    c10::SmallVector<int64_t, op_infer::SIZE> offsets_size = {0, 0};
    at::IntArrayRef offsets = at::IntArrayRef(offsets_size);
    auto keyCacheClone = key_cache.clone(at::MemoryFormat::Contiguous);
    auto valueCacheClone = value_cache.clone(at::MemoryFormat::Contiguous);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnScatterPaKvCache, key, keyCacheClone, slot_mapping, value,
        valueCacheClone, compress_lens, compress_seq_offsets, seq_lens, cache_mode, scatter_mode, strides, offsets);
    return std::make_tuple(keyCacheClone, valueCacheClone);
}

void npu_scatter_pa_kv_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    const c10::optional<at::Tensor>& compress_lens,
    const c10::optional<at::Tensor>& compress_seq_offsets,
    const c10::optional<at::Tensor>& seq_lens)
{
    char* cache_mode = "PA_NZ";
    char* scatter_mode = "None";
    c10::SmallVector<int64_t, op_infer::SIZE> strides_size = {1, 1};
    at::IntArrayRef strides = at::IntArrayRef(strides_size);
    c10::SmallVector<int64_t, op_infer::SIZE> offsets_size = {0, 0};
    at::IntArrayRef offsets = at::IntArrayRef(offsets_size);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnScatterPaKvCache, key, key_cache, slot_mapping, value, value_cache,
        compress_lens, compress_seq_offsets, seq_lens, cache_mode, scatter_mode, strides, offsets);
}

}
