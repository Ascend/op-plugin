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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target)
{
    std::string reduction_str = "none";
    if (reduction == at::Reduction::Mean) {
        reduction_str = "batchmean";
    } else if (reduction == at::Reduction::Sum) {
        reduction_str = "sum";
    }
    at::Tensor result = reduction_str == "none" ?
        npu_preparation::apply_tensor(self) : npu_preparation::apply_tensor({}, self.options(), self);
    at_npu::native::OpCommand cmd;
    cmd.Name("KLDiv")
        .Input(self)
        .Input(target)
        .Output(result)
        .Attr("reduction", reduction_str)
        .Attr("log_target", log_target)
        .Run();
    if (reduction == at::Reduction::Mean) {
        auto input_shape = self.sizes();
        int batch_square_size = c10::multiply_integers(input_shape) / input_shape[0];
        result.div_(batch_square_size);
    }
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor npu_kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target)
{
    std::string reduction_str = "none";
    if (reduction == at::Reduction::Mean) {
        reduction_str = "batchmean";
    } else if (reduction == at::Reduction::Sum) {
        reduction_str = "sum";
    }
    at::Tensor result = reduction_str == "none" ?
        npu_preparation::apply_tensor(self) : npu_preparation::apply_tensor({}, self.options(), self);
    at_npu::native::OpCommand cmd;
    cmd.Name("KLDiv")
        .Input(self)
        .Input(target)
        .Output(result)
        .Attr("reduction", reduction_str)
        .Attr("log_target", log_target)
        .Run();
    if (reduction == at::Reduction::Mean) {
        auto input_shape = self.sizes();
        int batch_square_size = input_shape.size() > 1 ? c10::multiply_integers(input_shape.slice(1)) : 1;
        result.div_(batch_square_size);
    }
    return result;
}

at::Tensor kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target)
{
    return npu_kl_div(self, target, reduction, log_target);
}
#endif

} // namespace acl_op
