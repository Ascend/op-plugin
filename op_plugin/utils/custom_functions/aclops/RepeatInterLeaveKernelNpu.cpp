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

namespace {
at::Tensor& repeat_interleave_out_nocheck(at::Tensor& result, const at::Tensor& self, int64_t repeats, int64_t dim)
{
    at::Scalar repeat = repeats;
    at_npu::native::OpCommand cmd;
    cmd.Name("RepeatInterleave")
        .Input(self)
        .Input(repeat, at::kLong)
        .Output(result)
        .Attr("axis", dim)
        .Run();

    return result;
}

at::Tensor& repeat_interleave_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& repeats,
                                          int64_t dim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("RepeatInterleave")
        .Input(self)
        .Input(repeats)
        .Output(result)
        .Attr("axis", dim)
        .Run();

    return result;
}

void check_dim_valid(int64_t real_dim, int64_t self_dim)
{
    int64_t dim_min = std::min(-self_dim, self_dim - 1);
    int64_t dim_max = std::max(-self_dim, self_dim - 1);
    TORCH_CHECK(
        (real_dim >= dim_min) && (real_dim <= dim_max),
        "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.",
        OPS_ERROR(ErrCode::VALUE));
}
} // namespace

at::Tensor repeat_interleave_common_nocheck(
    const at::Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim)
{
    int64_t real_dim = dim.value_or(0);
    int64_t self_dim = self.dim();
    check_dim_valid(real_dim, self_dim);

    TORCH_CHECK(
        repeats >= 1,
        "repeats can not be negative.",
        OPS_ERROR(ErrCode::VALUE));
    at::Tensor self_tensor = self;
    if (!dim.has_value()) {
        self_tensor = at::flatten(self_tensor);
    }
    if (repeats == 1) {
        return self_tensor;
    }

    auto op_infer_output_size = op_infer::repeat_interleave_npu_output_size(self_tensor, repeats, real_dim);
    at::Tensor result = npu_preparation::apply_tensor_with_format(self_tensor, op_infer_output_size, ACL_FORMAT_ND);
    repeat_interleave_out_nocheck(result, self_tensor, repeats, real_dim);

    return result;
}

at::Tensor repeat_interleave_common_nocheck(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim)
{
    int64_t real_dim = dim.value_or(0);
    int64_t self_dim = self.dim();
    check_dim_valid(real_dim, self_dim);

    at::Tensor self_tensor = self;
    at::Tensor repeats_tensor = repeats;
    if (repeats.dim() == 0) {
        repeats_tensor.unsqueeze_(0);
    }
    if (!dim.has_value()) {
        self_tensor = at::flatten(self_tensor);
    }

    TORCH_CHECK(
        (repeats.size(0) == self_tensor.size(real_dim)) || (repeats.size(0) == 1),
        "repeats must have the same size as input along dim.", OPS_ERROR(ErrCode::VALUE));

    repeats_tensor = acl_op::npu_dtype_cast(repeats_tensor, at::ScalarType::Float);
    auto op_infer_output_size = op_infer::repeat_interleave_npu_output_size(self_tensor, repeats_tensor, real_dim);

    at::Tensor result = npu_preparation::apply_tensor_with_format(self_tensor, op_infer_output_size, ACL_FORMAT_ND);
    repeat_interleave_out_nocheck(result, self_tensor, repeats, real_dim);
    return result;
}
} // namespace acl_op
