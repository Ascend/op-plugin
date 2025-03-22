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
at::Tensor &smooth_l1_loss_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &target,
                                           int64_t reduction, double beta)
{
    if (self.numel() == 0) {
        // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat).fill_(NAN);
        return result;
    }

    string reduction_str(op_plugin::utils::get_reduction_str(reduction));
    at_npu::native::OpCommand cmd;
    cmd.Name("SmoothL1LossV2")
        .Input(self)
        .Input(target)
        .Output(result)
        .Attr("reduction", reduction_str)
        .Attr("sigma", static_cast<float>(beta))
        .Run();
    return result;
}
} // namespace

at::Tensor &smooth_l1_loss_out(const at::Tensor &self, const at::Tensor &target, int64_t reduction, double beta,
                               at::Tensor &out)
{
    auto output_size = op_infer::smooth_l1_loss_npu_output_size(self, reduction);
    npu_preparation::CheckOut({self, target}, out, npu_preparation::get_tensor_npu_format(self), self.scalar_type(),
                              output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        smooth_l1_loss_out_npu_nocheck(contiguous_result, self, target, reduction, beta);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        smooth_l1_loss_out_npu_nocheck(out, self, target, reduction, beta);
    }
    return out;
}

at::Tensor smooth_l1_loss(const at::Tensor &self, const at::Tensor &target, int64_t reduction, double beta)
{
    auto output_size = op_infer::smooth_l1_loss_npu_output_size(self, reduction);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    smooth_l1_loss_out_npu_nocheck(result, self, target, reduction, beta);
    return result;
}
} // namespace acl_op
