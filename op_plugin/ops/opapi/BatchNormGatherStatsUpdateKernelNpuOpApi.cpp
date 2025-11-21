// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

std::tuple<at::Tensor &, at::Tensor &> batch_norm_gather_stats_update_npu_impl(const at::Tensor &self, const at::Tensor &sum,
                                                                               const at::Tensor &square_sum, const at::Tensor &counts, const at::Tensor &running_mean,
                                                                               const at::Tensor &running_var, double momentum, double eps,
                                                                               at::Tensor &batch_mean, at::Tensor &batch_invstd)
{
    at::Tensor counts_cp =
        counts.scalar_type() == at::kInt ? counts : at_npu::native::custom_ops::_npu_dtype_cast(counts, at::kInt);

    auto running_mean_dtype = running_mean.scalar_type();
    at::Tensor running_mean_ = at_npu::native::custom_ops::_npu_dtype_cast(
        at_npu::native::custom_ops::npu_format_cast(
            (running_mean.defined() ? running_mean : at::zeros({self.size(1)}, sum.options())), ACL_FORMAT_ND),
        sum.scalar_type());
    at::Tensor running_var_ = at_npu::native::custom_ops::_npu_dtype_cast(
        at_npu::native::custom_ops::npu_format_cast(
            (running_var.defined() ? running_var : at::ones({self.size(1)}, sum.options())), ACL_FORMAT_ND),
        sum.scalar_type());
    float momentumFloat = static_cast<float>(momentum);
    float epsFloat = static_cast<float>(eps);
    EXEC_NPU_CMD(aclnnSyncBatchNormGatherStats, sum, square_sum, counts, running_mean_, running_var_, momentumFloat, epsFloat, batch_mean, batch_invstd);
    
    if (running_mean.defined()) {
        if (running_mean_.scalar_type() != running_mean_dtype) {
            running_mean_ = at_npu::native::custom_ops::_npu_dtype_cast(running_mean_, running_mean_dtype);
            running_var_ = at_npu::native::custom_ops::_npu_dtype_cast(running_var_, running_mean_dtype);
        }
        running_mean.copy_(running_mean_);
        running_var.copy_(running_var_);
    }
    return std::tie(batch_mean, batch_invstd);
}

std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_update(const at::Tensor &self, const at::Tensor &sum,
                                                                  const at::Tensor &square_sum,
                                                                  const c10::optional<at::Tensor> &running_mean_opt,
                                                                  const c10::optional<at::Tensor> &running_var_opt,
                                                                  double momentum, double eps, const at::Tensor &counts)
{
    TORCH_CHECK(self.dim() > 1, "The dim input tensor [self] must more than 1." + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> output_size = {self.size(1)};

    DO_COMPATIBILITY(aclnnSyncBatchNormGatherStats, acl_op::batch_norm_gather_stats_update(self, sum, square_sum,
        running_mean_opt, running_var_opt, momentum, eps, counts));

    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        return acl_op::batch_norm_gather_stats_update(self, sum, square_sum, running_mean_opt, running_var_opt, momentum, eps, counts);
    }

    const at::Tensor &running_mean = c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
    const at::Tensor &running_var = c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

    at::Tensor batch_mean = npu_preparation::apply_tensor_without_format(output_size, self.options());
    at::Tensor batch_invstd = npu_preparation::apply_tensor_without_format(output_size, self.options());

    batch_norm_gather_stats_update_npu_impl(self, sum, square_sum, counts, running_mean, running_var, momentum, eps, batch_mean, batch_invstd);
    return std::make_tuple(batch_mean, batch_invstd);
}
}