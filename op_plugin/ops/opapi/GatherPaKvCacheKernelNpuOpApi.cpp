// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using tensor_list = std::tuple<at::Tensor, at::Tensor>;

void npu_gather_pa_kv_cache(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& seq_lens,
    at::Tensor& key,
    at::Tensor& value,
    const c10::optional<at::Tensor>& seq_offset,
    bool is_seq_lens_cumsum)
{
    int64_t cache_format = at_npu::native::custom_ops::get_npu_format(key_cache);
    const char* cacheMode = (cache_format == ACL_FORMAT_FRACTAL_NZ) ? "PA_NZ" : "Norm";
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnGatherPaKvCache, key_cache, value_cache, block_tables, seq_lens, key, value,
        seq_offset, cacheMode, is_seq_lens_cumsum);
}

tensor_list npu_gather_pa_kv_cache_functional(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& seq_lens,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& seq_offset,
    bool is_seq_lens_cumsum)
{
    at::Tensor key_clone = key.clone();
    at::Tensor value_clone = value.clone();
    int64_t cache_format = at_npu::native::custom_ops::get_npu_format(key_cache);
    const char* cacheMode = (cache_format == ACL_FORMAT_FRACTAL_NZ) ? "PA_NZ" : "Norm";
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnGatherPaKvCache, key_cache, value_cache, block_tables, seq_lens, key_clone, value_clone,
        seq_offset, cacheMode, is_seq_lens_cumsum);

    return std::tuple<at::Tensor, at::Tensor>(key_clone, value_clone);
}
}