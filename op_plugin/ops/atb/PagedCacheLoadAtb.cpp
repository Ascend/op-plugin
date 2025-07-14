// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

namespace atb {
std::tuple<at::Tensor &, at::Tensor &> npu_paged_cache_load_out(
    const at::Tensor &key_cache, const at::Tensor &value_cache,
    const at::Tensor &block_table, const at::Tensor &context_lens,
    const c10::optional<at::Tensor> &seq_starts, bool cumsum, at::Tensor &key,
    at::Tensor &value)
{
    auto key_cache_format = at_npu::native::get_npu_format(key_cache);
    bool has_seq_starts =
        seq_starts.has_value() && seq_starts.value().defined();
    int8_t kv_cache_cfg = key_cache_format == aclFormat::ACL_FORMAT_FRACTAL_NZ ? 0 : 1;

    const c10::OptionalDeviceGuard device_guard(device_of(key_cache));

    EXEC_ATB_CMD(AtbPagedCacheLoad, key_cache, value_cache, block_table,
                 context_lens, key, value, seq_starts, kv_cache_cfg, cumsum,
                 has_seq_starts);

    return std::forward_as_tuple(key, value);
}

std::tuple<at::Tensor, at::Tensor> npu_paged_cache_load(
    const at::Tensor &key_cache, const at::Tensor &value_cache,
    const at::Tensor &block_table, const at::Tensor &context_lens,
    const c10::optional<at::Tensor> &seq_starts, bool cumsum)
{
    int32_t num_tokens = 0;
    if (cumsum) {
        num_tokens =
            context_lens.numel() > 0 ? context_lens[-1].item<int32_t>() : 0;
    } else {
        for (int i = 0; i < context_lens.numel(); i++) {
            num_tokens += context_lens[i].item<int32_t>();
        }
    }
    at::Tensor key =
        at::empty({num_tokens, key_cache.size(2), key_cache.size(3)},
                  key_cache.options());
    at::Tensor value =
        at::empty({num_tokens, value_cache.size(2), value_cache.size(3)},
                  value_cache.options());

    auto key_cache_format = at_npu::native::get_npu_format(key_cache);
    bool has_seq_starts =
        seq_starts.has_value() && seq_starts.value().defined();
    int8_t kv_cache_cfg = key_cache_format == aclFormat::ACL_FORMAT_FRACTAL_NZ ? 0 : 1;

    const c10::OptionalDeviceGuard device_guard(device_of(key_cache));

    EXEC_ATB_CMD(AtbPagedCacheLoad, key_cache, value_cache, block_table,
                 context_lens, key, value, seq_starts, kv_cache_cfg, cumsum,
                 has_seq_starts);

    return std::make_tuple(std::move(key), std::move(value));
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m) {
    m.def(
        "npu_paged_cache_load.out(Tensor key_cache, Tensor value_cache, Tensor "
        "block_table, Tensor context_lens, *, Tensor? seq_starts=None, bool "
        "cumsum=False, Tensor(a!) key, Tensor(b!) value) -> (Tensor(a!), Tensor(b!))");

    m.def(
        "npu_paged_cache_load(Tensor key_cache, Tensor value_cache, Tensor "
        "block_table, Tensor context_lens, *, Tensor? seq_starts=None, bool "
        "cumsum=False) -> (Tensor, Tensor)");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m) {
    m.impl("npu_paged_cache_load.out", TORCH_FN(atb::npu_paged_cache_load_out));
    m.impl("npu_paged_cache_load", TORCH_FN(atb::npu_paged_cache_load));
}
}  // namespace
}  // namespace atb