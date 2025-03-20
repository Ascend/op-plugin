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
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> lerp_broadcast_size(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight)
{
    auto expanded_size = op_infer::broadcast_ops_npu_output_size(self, end);
    auto output_size = op_infer::broadcast_ops_npu_output_size(expanded_size, weight.sizes());
    return output_size;
}

at::Tensor& lerp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Lerp")
        .Input(self)
        .Input(end)
        .Input(weight)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& lerp_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& end,
    at::Scalar weight)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Lerp")
        .Input(self)
        .Input(end)
        .Input(weight, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& lerp_out(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight,
    at::Tensor& out)
{
    auto output_size = lerp_broadcast_size(self, end, weight);
    npu_preparation::CheckOut(
        {self, end, weight},
        out,
        self,
        output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        lerp_out_npu_nocheck(contiguous_result, self, end, weight);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        lerp_out_npu_nocheck(out, self, end, weight);
    }
    return out;
}

at::Tensor& lerp_out(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight,
    at::Tensor& out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, end);
    npu_preparation::CheckOut(
        {self, end},
        out,
        self,
        output_size);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        lerp_out_npu_nocheck(contiguous_result, self, end, weight);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        lerp_out_npu_nocheck(out, self, end, weight);
    }
    return out;
}

at::Tensor lerp(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight)
{
    auto output_size = lerp_broadcast_size(self, end, weight);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    lerp_out_npu_nocheck(result, self, end, weight);
    return result;
}

at::Tensor lerp(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, end);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    lerp_out_npu_nocheck(result, self, end, weight);
    return result;
}

at::Tensor& lerp_(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight)
{
    c10::SmallVector<int64_t, SIZE> self_size = op_infer::array_to_small_vector(self.sizes());
    auto output_size = lerp_broadcast_size(self, end, weight);
    TORCH_CHECK(self_size == output_size,
        "output with shape ", self_size, " doesn't match the broadcast shape ", output_size,
        OPS_ERROR(ErrCode::PARAM));
    return acl_op::lerp_out(self, end, weight, self);
}

at::Tensor& lerp_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight)
{
    c10::SmallVector<int64_t, SIZE> self_size = op_infer::array_to_small_vector(self.sizes());
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, end);
    TORCH_CHECK(self_size == output_size,
        "output with shape ", self_size, " doesn't match the broadcast shape ", output_size,
        OPS_ERROR(ErrCode::PARAM));
    return acl_op::lerp_out(self, end, weight, self);
}
} // namespace acl_op
