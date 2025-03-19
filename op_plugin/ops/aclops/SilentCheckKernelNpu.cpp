// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace {
at::Tensor silent_check_nocheck(at::Tensor &input_grad, const at::Tensor &val, at::Tensor &pre_val, at::Tensor &min_val,
                                at::Tensor &max_val, const at::Tensor &val_counter, int64_t c_min_steps,
                                double c_thresh_l1, double c_coeff_l1, double c_thresh_l2, double c_coeff_l2,
                                at::Tensor &result)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SilentCheck")
        .Input(val)
        .Input(input_grad)
        .Input(pre_val)
        .Input(min_val)
        .Input(max_val)
        .Input(val_counter)
        .Output(input_grad)
        .Output(pre_val)
        .Output(min_val)
        .Output(max_val)
        .Output(result)
        .Attr("c_min_steps", c_min_steps)
        .Attr("c_thresh_l1", static_cast<float>(c_thresh_l1))
        .Attr("c_coeff_l1", static_cast<float>(c_coeff_l1))
        .Attr("c_thresh_l2", static_cast<float>(c_thresh_l2))
        .Attr("c_coeff_l2", static_cast<float>(c_coeff_l2))
        .Run();
    return result;
}
} // namespace

at::Tensor _npu_silent_check(at::Tensor &input_grad, const at::Tensor &val, at::Tensor &pre_val,
                             at::Tensor &min_val, at::Tensor &max_val, const at::Tensor &val_counter,
                             int64_t c_min_steps, double c_thresh_l1, double c_coeff_l1, double c_thresh_l2,
                             double c_coeff_l2)
{
    at::Tensor result = npu_preparation::apply_tensor(val_counter);
    return silent_check_nocheck(input_grad, val, pre_val, min_val, max_val, val_counter, c_min_steps, c_thresh_l1,
                                c_coeff_l1, c_thresh_l2, c_coeff_l2, result);
}
} // namespace acl_op
