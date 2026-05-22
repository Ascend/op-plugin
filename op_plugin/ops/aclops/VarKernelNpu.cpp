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
#include "op_plugin/utils/StdVarCorrectionUtils.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result)
{
    if (!correction_fits_aclnn_int64(correction)) {
        at::Tensor cpu_out = result.cpu();
        at::var_out(cpu_out, self.cpu(), dim, correction, keepdim);
        result.copy_(cpu_out);
        return result;
    }
    bool unbiased = !(correction.has_value() && correction.value().toLong() == 0);
    int64_t real_correction = correction.has_value() ? correction.value().toLong() : 1;
    return cal_var_out(self, dim.value_or(at::IntArrayRef{}), real_correction, unbiased, keepdim, result);
}

at::Tensor var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim)
{
    if (!correction_fits_aclnn_int64(correction)) {
        return at::var(self.cpu(), dim, correction, keepdim).to(self.options());
    }
    bool unbiased = !(correction.has_value() && correction.value().toLong() == 0);
    int64_t real_correction = correction.has_value() ? correction.value().toLong() : 1;
    return cal_var(self, dim.value_or(at::IntArrayRef{}), real_correction, unbiased, keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim)
{
    if (!correction_fits_aclnn_int64(correction)) {
        auto cpu_tup = at::var_mean(self.cpu(), dim, correction, keepdim);
        return std::make_tuple(
            std::get<0>(cpu_tup).to(self.options()),
            std::get<1>(cpu_tup).to(self.options()));
    }
    bool unbiased = !(correction.has_value() && correction.value().toLong() == 0);
    int64_t real_correction = correction.has_value() ? correction.value().toLong() : 1;
    return cal_var_mean(self, dim.value_or(at::IntArrayRef{}), unbiased, real_correction, keepdim);
}
} // namespace acl_op
