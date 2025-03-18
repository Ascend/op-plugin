// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &batch_norm_elemt_nocheck(at::Tensor &result, const at::Tensor &self,
                                     const c10::optional<at::Tensor> &weight_opt,
                                     const c10::optional<at::Tensor> &bias_opt, const at::Tensor &mean,
                                     const at::Tensor &invstd, double eps)
{
    TORCH_CHECK(self.dim() >= 2, "self's dim must be greater than or equal to 2, but got ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    auto dim_c = self.size(1);
    auto options = self.options().dtype(at::kFloat);
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor &weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });

    at::Tensor weight_val = weight.defined() ? weight : at::ones({dim_c}, options);
    at::Tensor bias_val = bias.defined() ? bias : at::ones({dim_c}, options);
    at::Tensor mean_val = mean.defined() ? mean : at::ones({dim_c}, options);
    at::Tensor invstd_val = invstd.defined() ? invstd : at::ones({dim_c}, options);
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && mean.dim() == 1 && invstd.dim() == 1,
        "weight, bias, mean, invstd: must be only one dimension." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.size(0) == dim_c && bias.size(0) == dim_c && mean.size(0) == dim_c && invstd.size(0) == dim_c,
        "weight, bias, mean, invstd: shape must be equal to  self's dim_c." + OPS_ERROR(ErrCode::PARAM));
    at::Tensor one = at::ones({1}, options);
    auto variance = at::mul(invstd_val, invstd_val);
    variance = at::div(one, variance) - eps;
    int64_t self_dim = self.dim();
    at::Tensor self_5d(self);
    c10::SmallVector<int64_t, N> self_shape = op_infer::array_to_small_vector(self.sizes());
    if (self_dim > 5) {
        self_5d = self.reshape({self.size(0), self.size(1), self.size(2), self.size(3), -1});
    }
    std::tuple<at::Tensor, at::Tensor, at::Tensor> bn_reult =
        at::native_batch_norm(self_5d, weight_val, bias_val, mean_val, variance, false, 0.0, eps);
    result.copy_(std::get<0>(bn_reult));
    if (self_dim > 5) {
        result = result.view(self_shape);
    }
    return result;
}
} // namespace

at::Tensor &batch_norm_elemt_out(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                                 const c10::optional<at::Tensor> &bias, const at::Tensor &mean,
                                 const at::Tensor &invstd, double eps, at::Tensor &out)
{
    const at::Tensor &bias_value = c10::value_or_else(bias, [] { return at::Tensor(); });
    const at::Tensor &weight_value = c10::value_or_else(weight, [] { return at::Tensor(); });

    npu_preparation::CheckOut({input, bias_value, weight_value}, out, input);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        batch_norm_elemt_nocheck(contiguous_result, input, weight_value, bias_value, mean, invstd, eps);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        batch_norm_elemt_nocheck(out, input, weight_value, bias_value, mean, invstd, eps);
    }

    return out;
}

at::Tensor batch_norm_elemt(const at::Tensor &input, const c10::optional<at::Tensor> &weight,
                            const c10::optional<at::Tensor> &bias, const at::Tensor &mean, const at::Tensor &invstd,
                            double eps)
{
    at::Tensor result = npu_preparation::apply_tensor(input);
    batch_norm_elemt_nocheck(result, input, weight, bias, mean, invstd, eps);
    return result;
}
} // namespace acl_op
