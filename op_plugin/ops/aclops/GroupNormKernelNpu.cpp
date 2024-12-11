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
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
inline bool is_support_group_norm_swish(const at::Tensor& X)
{
    // GroupNorm supports format NC1HWC0
    if (npu_preparation::get_tensor_npu_format(X) == ACL_FORMAT_NC1HWC0) {
        return false;
    }

    if ((c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend310P1 &&
         c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
        (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4)) {
        return true;
    }

    return false;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_out_npu(
    at::Tensor& y,
    at::Tensor& mean,
    at::Tensor& variance,
    at::Tensor& rstd,
    const at::Tensor& X,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double eps,
    int64_t C)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GroupNorm")
        .Input(X)
        .Input(gamma)
        .Input(beta)
        .Output(y)
        .Output(mean)
        .Output(variance)
        .Attr("num_groups", num_groups)
        .Attr("eps", static_cast<float>(eps))
        .Attr("is_training", true)
        .Run();

    rstd = 1.0 / (variance + eps).sqrt();
    return std::make_tuple(y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_swish_out_npu(
    at::Tensor& y,
    at::Tensor& mean,
    at::Tensor& variance,
    at::Tensor& rstd,
    const at::Tensor& X,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double eps,
    int64_t C)
{
    at_npu::native::OpCommand cmd;
    if (is_support_group_norm_swish(X)) {
        cmd.Name("GroupNormSwish")
            .Input(X)
            .Input(gamma)
            .Input(beta)
            .Output(y)
            .Output(mean)
            .Output(rstd)
            .Attr("num_groups", num_groups)
            .Attr("eps", static_cast<float>(eps))
            .Attr("activate_swish", false)
            .Run();
        return std::make_tuple(y, mean, rstd);
    }
    return native_group_norm_out_npu(y, mean, variance, rstd, X, gamma, beta, num_groups, eps, C);
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt,
    const c10::optional<at::Tensor>& beta_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps)
{
    at::Tensor result = npu_preparation::ApplyTensor(X);
    at::Tensor mean = npu_preparation::apply_tensor_with_format({N, group}, X.options(), ACL_FORMAT_ND);
    at::Tensor variance = npu_preparation::apply_tensor_with_format({N, group}, X.options(), ACL_FORMAT_ND);
    at::Tensor rstd = npu_preparation::apply_tensor_with_format({N, group}, X.options(), ACL_FORMAT_ND);
    const at::Tensor& gamma_ = c10::value_or_else(gamma_opt, [] {return at::Tensor();});
    at::Tensor gamma = gamma_.defined() ? gamma_ : at::ones({C}, X.options());
    const at::Tensor& beta_ = c10::value_or_else(beta_opt, [] {return at::Tensor();});
    at::Tensor beta = beta_.defined() ? beta_ : at::zeros({C}, X.options());
    // DO_COMPATIBILITY is used to determine whether GroupNormSwish exists
    DO_COMPATIBILITY(aclnnGroupNormSwish, acl_op::native_group_norm_out_npu(result, mean, variance, rstd, X, gamma,
                                                                            beta, group, eps, C));
    return native_group_norm_swish_out_npu(result, mean, variance, rstd, X, gamma, beta, group, eps, C);
}
} // namespace acl_op
