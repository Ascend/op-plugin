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
using tensor_list1 = std::tuple<at::Tensor &, at::Tensor &>;
using tensor_list2 = std::tuple<at::Tensor, at::Tensor>;

namespace {
tensor_list1 batch_norm_gather_stats_with_counts_npu_impl(at::Tensor &mean_all, at::Tensor &invstd_all,
                                                          const at::Tensor &self, const at::Tensor &mean,
                                                          const at::Tensor &invstd, const at::Tensor &running_mean,
                                                          const at::Tensor &running_var, double momentum, double eps,
                                                          const at::Tensor &counts)
{
    auto options = self.options();
    TORCH_CHECK(self.dim() > 1, "The dim input tensor [self] must more than 1." + OPS_ERROR(ErrCode::PARAM));
    auto dim_c = self.size(1);
    at::Tensor mean_cp = at_npu::native::custom_ops::npu_dtype_cast(mean, at::kFloat);
    at::Tensor invstd_cp = at_npu::native::custom_ops::npu_dtype_cast(invstd, at::kFloat);
    auto running_mean_dtype = running_mean.scalar_type();
    at::Tensor running_mean_val = at_npu::native::custom_ops::npu_dtype_cast(
        at_npu::native::custom_ops::npu_format_cast(
            (running_mean.defined() ? running_mean.unsqueeze(0) : at::zeros({1, dim_c}, options)), ACL_FORMAT_ND),
        at::kFloat);
    at::Tensor running_var_val = at_npu::native::custom_ops::npu_dtype_cast(
        at_npu::native::custom_ops::npu_format_cast(
            (running_var.defined() ? running_var.unsqueeze(0) : at::ones({1, dim_c}, options)), ACL_FORMAT_ND),
        at::kFloat);
    std::vector<int64_t> axes = {0};
    at::Tensor counts_tensor = at_npu::native::custom_ops::npu_dtype_cast(counts, mean_cp.scalar_type());
    at::Tensor counts_tensor_t = counts_tensor.unsqueeze(-1);
    at::Tensor counts_tensor_broadcast = acl_op::npu_broadcast(counts_tensor_t, invstd.sizes());
    at::Tensor counts_all_sum = npu_preparation::apply_tensor_with_sizes({1, dim_c}, mean_cp.options());
    at_npu::native::OpCommand cmd_reduce;
    cmd_reduce.Name("ReduceSum")
        .Input(counts_tensor_broadcast)
        .Input(axes, at::kInt)
        .Attr("keep_dims", true)
        .Output(counts_all_sum)
        .Run();

    at::Tensor counts_all_sum_broadcast = counts_all_sum.expand(counts_tensor_broadcast.sizes());
    at_npu::native::OpCommand cmd_mean;
    cmd_mean.Name("ReduceMeanWithCount")
        .Input(mean_cp)
        .Input(counts_tensor_broadcast)
        .Input(counts_all_sum_broadcast)
        .Output(mean_all)
        .Attr("axes", axes)
        .Attr("keep_dims", true)
        .Run();

    at::Tensor mean_broadcast = mean_all.expand(mean.sizes());
    at_npu::native::OpCommand cmd_batch;
    cmd_batch.Name("SyncBatchNormGatherStatsWithCounts")
        .Input(mean_cp)
        .Input(invstd_cp)
        .Input(counts_tensor_broadcast)
        .Input(mean_broadcast)
        .Input(counts_all_sum)
        .Input(running_var_val)
        .Output(invstd_all)
        .Output(running_var_val)
        .Attr("momentum", static_cast<float>(momentum))
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    if (running_mean.defined()) {
        at_npu::native::OpCommand cmd_sync;
        cmd_sync.Name("SyncBNTrainingUpdate")
            .Input(mean_all)
            .Input(running_mean_val)
            .Output(running_mean_val)
            .Attr("momentum", static_cast<float>(momentum))
            .Run();
        // running_mean almost apply is the same as running_var
        if (running_mean_val.scalar_type() != running_mean_dtype) {
            running_mean_val = at_npu::native::custom_ops::npu_dtype_cast(running_mean_val, running_mean_dtype);
            running_var_val = at_npu::native::custom_ops::npu_dtype_cast(running_var_val, running_mean_dtype);
        }
        running_mean.copy_(running_mean_val.squeeze(0));
        running_var.copy_(running_var_val.squeeze(0));
    }

    return std::tie(mean_all, invstd_all);
}
} // namespace

tensor_list2 batch_norm_gather_stats_with_counts(const at::Tensor &input, const at::Tensor &mean,
                                                 const at::Tensor &invstd,
                                                 const c10::optional<at::Tensor> &running_mean,
                                                 const c10::optional<at::Tensor> &running_var, double momentum,
                                                 double eps, const at::Tensor &counts)
{
    const at::Tensor &running_mean_opt = c10::value_or_else(running_mean, [] { return at::Tensor(); });
    const at::Tensor &running_var_opt = c10::value_or_else(running_var, [] { return at::Tensor(); });
    bool is_fully_fp16 = false;
    if (input.scalar_type() == mean.scalar_type() && input.scalar_type() == at::kHalf) {
        is_fully_fp16 = true;
    }

    at::Tensor mean_all = npu_preparation::apply_tensor({1, input.size(1)}, input.options().dtype(at::kFloat), input);
    at::Tensor invstd_all = npu_preparation::apply_tensor({1, input.size(1)}, input.options().dtype(at::kFloat), input);

    batch_norm_gather_stats_with_counts_npu_impl(mean_all, invstd_all, input, mean, invstd, running_mean_opt, running_var_opt,
                                                 momentum, eps, counts);

    if (is_fully_fp16) {
        mean_all = at_npu::native::custom_ops::npu_dtype_cast(mean_all, at::kHalf);
        invstd_all = at_npu::native::custom_ops::npu_dtype_cast(invstd_all, at::kHalf);
    }

    return std::make_tuple(mean_all.squeeze(0), invstd_all.squeeze(0));
}
} // namespace acl_op
