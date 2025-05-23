// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor _cdist_forward(const at::Tensor &x1, const at::Tensor &x2, const double p,
                          c10::optional<int64_t> compute_mode)
{
    TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1),
        " X2: ", x2.size(-1),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(at::isFloatingType(x1.scalar_type()),
        "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(at::isFloatingType(x1.scalar_type()),
        "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(p >= 0, "cdist only supports non-negative p values" + OPS_ERROR(ErrCode::PARAM));

    // Since double is not supported in NPU, the type of P needs to be converted from double to float.
    float p_float;
    if (std::isinf(p)) {
        p_float = -1;
    } else {
        TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu does not support float64"
            + OPS_ERROR(ErrCode::TYPE));
        p_float = static_cast<float>(p);
    }

    int64_t mode = compute_mode.value_or(0);
    TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode,
        OPS_ERROR(ErrCode::VALUE));

    // Broadcast
    int64_t c1 = x1.size(-1);
    int64_t c2 = x2.size(-1);
    int64_t r1 = x1.size(-2);
    int64_t r2 = x2.size(-2);
    auto dim1 = x1.dim();
    auto dim2 = x2.dim();

    at::IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
    at::IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
    std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

    int expand_batch_product =
        std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(), 1, std::multiplies<int64_t>());
    std::vector<int64_t> tensor1_view{expand_batch_product, r1, 1, c1};
    std::vector<int64_t> tensor2_view{expand_batch_product, 1, r2, c2};
    std::vector<int64_t> result_size{expand_batch_product, r1, r2};
    std::vector<int64_t> tensor_broadcast_size = at::infer_size(tensor1_view, tensor2_view);

    // Broadcast batch dim.
    at::Tensor tensor1_expanded = x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
    at::Tensor tensor2_expanded = x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);

    // Broadcast r1 and r2.
    at::Tensor tensor1_broadcast = tensor1_expanded.expand(tensor_broadcast_size).contiguous();
    at::Tensor tensor2_broadcast = tensor2_expanded.expand(tensor_broadcast_size).contiguous();

    auto output_size = op_infer::cdist_npu_output_size(x1, x2);
    at::Tensor result = npu_preparation::apply_tensor(tensor1_broadcast, result_size);

    at_npu::native::OpCommand cmd;
    cmd.Name("Cdist").Input(tensor1_broadcast).Input(tensor2_broadcast).Attr("p", p_float).Output(result).Run();
    return result.view(output_size);
}

at::Tensor cdist(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    if (x1.has_names() || x2.has_names()) {
        auto maybe_outnames = at::namedinference::compute_cdist_outnames(x1, x2);
        // aten::view do not support named tensor now,
        // therefore we have to drop names for x1 and x2
        at::Tensor x1_no_name = npu_preparation::apply_tensor(x1);
        at::Tensor x2_no_name = npu_preparation::apply_tensor(x2);
        auto result = at::_cdist_forward(x1_no_name, x2_no_name, p, compute_mode);
        at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
        return result;
    }

    return at::_cdist_forward(x1, x2, p, compute_mode);
}
} // namespace acl_op
