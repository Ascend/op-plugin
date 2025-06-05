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
std::unordered_map<c10::string_view, int> activation_type_map = {
    {"activation_undefined", 0},
    {"activation_relu", 1},
    {"activation_gelu", 2},
    {"activation_fast_gelu", 3},
    {"activation_swish", 4},
    {"activation_log", 5},
    {"activation_swiglu_forward", 6},
    {"activation_swiglu_backward", 7},
    {"activation_sigmoid", 8},
    {"activation_faster_gelu_forward", 9},
    {"activation_max", 10}
};

std::tuple<int> get_fused_add_div_mode(c10::optional<c10::string_view> activation_type_opt)
{
    c10::string_view activation_type_str = activation_type_opt.value_or("activation_sigmoid");
    return std::make_tuple(activation_type_map[activation_type_str]);
}
}

std::tuple<at::Tensor&, at::Tensor&> npu_fused_add_topk_div_out(const at::Tensor &x, const at::Tensor &add_num, const c10::optional<at::Tensor> &mapping_num, const c10::optional<at::Tensor> &mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num, int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale, bool enable_expert_mapping,
    at::Tensor &y,
    at::Tensor &indices)
{
    const c10::OptionalDeviceGuard device_guard(device_of(x));
    float scale_float = static_cast<float>(scale);
    auto mode = get_fused_add_div_mode(activation_type_opt);
    int activation_type = std::get<0>(mode);
    EXEC_ATB_CMD(AtbFusedAddTopkDiv, x, add_num, mapping_num, mapping_table, group_num, group_topk, n, k, activation_type, is_norm, scale_float, enable_expert_mapping, y, indices);
    return std::forward_as_tuple(y, indices);
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_fused_add_topk_div.out(Tensor x, Tensor add_num, *, Tensor? mapping_num=None, Tensor? mapping_table=None, str? activation_type=None, int group_num=1, int group_topk=1, int n=1, int k=1,  bool is_norm=True, float scale=1, bool enable_expert_mapping=False, Tensor(a!) y, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_fused_add_topk_div.out", TORCH_FN(atb::npu_fused_add_topk_div_out));
}
}

} // namespace
