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

std::tuple<at::Tensor, at::Tensor> npu_random_choice_with_mask(
    const at::Tensor& x,
    int64_t count,
    int64_t seed,
    int64_t seed2)
{
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::Bool,
        "The input.dtype should be bool, but get",
        x.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        x.dim() <= 5 && x.dim() >= 1,
        "The input.dim should be in [1, 5], but get",
        x.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(count > 0, "The count must greater than 0, but get", count, OPS_ERROR(ErrCode::VALUE));

    at::Tensor result = npu_preparation::apply_tensor({count, x.dim()}, x.options().dtype(at::kInt), x);
    at::Tensor mask = npu_preparation::apply_tensor(x, {count});
    at_npu::native::OpCommand cmd;
    cmd.Name("RandomChoiceWithMask")
        .Input(x)
        .Output(result)
        .Output(mask)
        .Attr("count", count)
        .Attr("seed", seed)
        .Attr("seed2", seed2)
        .Run();

    return std::tie(result, mask);
}

} // namespace acl_op
