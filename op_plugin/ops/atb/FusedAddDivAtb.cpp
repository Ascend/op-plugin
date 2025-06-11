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
#include "op_plugin/utils/custom_functions/atb/Utils.h"

namespace atb {
namespace {
std::unordered_map<c10::string_view, int> activation_type_map = {
    {"activation_sigmoid", 8}
};

int get_fused_add_div_mode(c10::optional<c10::string_view> activation_type_opt)
{
    int activation_type = atb::utils::get_op_mode(
        activation_type_map, activation_type_opt, "activation_sigmoid", "activation_type");
    return activation_type;
}
}

std::tuple<at::Tensor, at::Tensor> npu_fused_add_topk_div(const at::Tensor &x, const at::Tensor &add_num, const c10::optional<at::Tensor> &mapping_num, const c10::optional<at::Tensor> &mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num, int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale, bool enable_expert_mapping)
{
    const c10::OptionalDeviceGuard device_guard(device_of(x));
    int64_t a = x.size(0);
    at::Tensor y = at::empty({a, k}, x.options().dtype(c10::ScalarType::Float));
    at::Tensor indices = at::empty({a, k}, x.options().dtype(c10::ScalarType::Int));
    float scale_float = static_cast<float>(scale);
    auto activation_type = get_fused_add_div_mode(activation_type_opt);
    EXEC_ATB_CMD(AtbFusedAddTopkDiv, x, add_num, mapping_num, mapping_table, group_num, group_topk, n, k, activation_type, is_norm, scale_float, enable_expert_mapping, y, indices);
    return std::make_tuple(y, indices);
}

std::tuple<at::Tensor&, at::Tensor&> npu_fused_add_topk_div_out(const at::Tensor &x, const at::Tensor &add_num, const c10::optional<at::Tensor> &mapping_num, const c10::optional<at::Tensor> &mapping_table,
    c10::optional<c10::string_view> activation_type_opt, int64_t group_num, int64_t group_topk, int64_t n, int64_t k, bool is_norm, double scale, bool enable_expert_mapping,
    at::Tensor &y,
    at::Tensor &indices)
{
    const c10::OptionalDeviceGuard device_guard(device_of(x));
    float scale_float = static_cast<float>(scale);
    auto activation_type = get_fused_add_div_mode(activation_type_opt);
    EXEC_ATB_CMD(AtbFusedAddTopkDiv, x, add_num, mapping_num, mapping_table, group_num, group_topk, n, k, activation_type, is_norm, scale_float, enable_expert_mapping, y, indices);
    return std::forward_as_tuple(y, indices);
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("npu_fused_add_topk_div(Tensor x, Tensor add_num, *, Tensor? mapping_num=None, Tensor? mapping_table=None, str? activation_type=None, int group_num=1, int group_topk=1, int n=1, int k=1,  bool is_norm=True, float scale=1, bool enable_expert_mapping=False) -> (Tensor, Tensor)");
    m.def("npu_fused_add_topk_div.out(Tensor x, Tensor add_num, *, Tensor? mapping_num=None, Tensor? mapping_table=None, str? activation_type=None, int group_num=1, int group_topk=1, int n=1, int k=1,  bool is_norm=True, float scale=1, bool enable_expert_mapping=False, Tensor(a!) y, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("npu_fused_add_topk_div", TORCH_FN(atb::npu_fused_add_topk_div));
    m.impl("npu_fused_add_topk_div.out", TORCH_FN(atb::npu_fused_add_topk_div_out));
}
}
}
