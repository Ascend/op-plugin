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

#include <acl/acl.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

namespace atb {
namespace {
std::unordered_map<c10::string_view, int> kernel_type_map = {
    {"kernel_type_default", 0},
    {"kernel_type_high_precision", 1}
};

std::unordered_map<c10::string_view, int> mask_type_map = {
    {"no_mask", 0},
    {"mask_type_triu", 1}
};

std::unordered_map<c10::string_view, int> input_layout_map = {
    {"type_bsnd", 0},
    {"type_bnsd", 1}
};

std::unordered_map<c10::string_view, int> calc_type_map = {
    {"calc_type_default", 0},
    {"calc_type_first_ring", 1},
    {"calc_type_max", 2}
};

std::tuple<int, int, int, int> get_ring_mode(c10::optional<c10::string_view> kernel_type_opt, c10::optional<c10::string_view> mask_type_opt, c10::optional<c10::string_view> input_layout_opt, c10::optional<c10::string_view> calc_type_opt)
{
    c10::string_view kernel_type_str = kernel_type_opt.value_or("kernel_type_high_precision");
    c10::string_view mask_type_str = mask_type_opt.value_or("no_mask");
    c10::string_view input_layout_str = input_layout_opt.value_or("type_bsnd");
    c10::string_view calc_type_str = calc_type_opt.value_or("calc_type_first_ring");
    return std::make_tuple(mask_type_map[kernel_type_str], mask_type_map[mask_type_str], input_layout_map[input_layout_str], calc_type_map[calc_type_str]);
}
}

std::tuple<at::Tensor&, at::Tensor&> npu_ring_mla_out(const at::Tensor &q_nope, const at::Tensor &q_rope, const at::Tensor &k_nope, const at::Tensor &k_rope, const at::Tensor &value,
    const at::Tensor &mask, const at::Tensor &seqlen, int64_t head_num, int64_t kv_head_num, const c10::optional<at::Tensor> &pre_out, const c10::optional<at::Tensor> &prev_lse,
    double qk_scale, c10::optional<c10::string_view> kernel_type_opt, c10::optional<c10::string_view> mask_type_opt,
    c10::optional<c10::string_view> input_layout_opt, c10::optional<c10::string_view> calc_type_opt,
    at::Tensor& output,
    at::Tensor& softmax_lse)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q_nope));
    float qkScale_float = static_cast<float>(qk_scale);
    auto mode = get_ring_mode(kernel_type_opt, mask_type_opt, input_layout_opt, calc_type_opt);
    int kernel_type = std::get<0>(mode);
    int mask_type = std::get<1>(mode);
    int input_layout = std::get<2>(mode);
    int calc_type = std::get<3>(mode);
    EXEC_ATB_CMD(AtbRingMLA, q_nope, q_rope, k_nope, k_rope, value, mask, seqlen, pre_out, prev_lse, head_num, kv_head_num, qkScale_float, kernel_type, mask_type, input_layout, calc_type, output, softmax_lse);
    return std::forward_as_tuple(output, softmax_lse);
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_ring_mla.out(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor value, Tensor mask, Tensor seqlen, int head_num, int kv_head_num, *, Tensor? pre_out=None, Tensor? prev_lse=None, float qk_scale=1, str? kernel_type=None, str? mask_type=None, str? input_layout=None, str? calc_type=None, Tensor(a!) output, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_ring_mla.out", TORCH_FN(atb::npu_ring_mla_out));
}
}

}
