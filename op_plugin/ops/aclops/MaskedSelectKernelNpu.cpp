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
std::tuple<at::Tensor, at::Tensor> expand_outplace_npu(const at::Tensor& to_expand1, const at::Tensor& to_expand2)
{
    if (to_expand1.sizes().equals(to_expand2.sizes())) {
        return std::make_tuple(to_expand1, to_expand2);
    }

    auto expanded_size = at::infer_size(to_expand1.sizes(), to_expand2.sizes());
    return std::make_tuple(to_expand1.expand(expanded_size, true), to_expand2.expand(expanded_size, true));
}

at::SmallVector<int64_t, SIZE> masked_select_npu_output_size(const at::Tensor& self, const at::Tensor& mask)
{
    at::Tensor mask_cast;
    at::Tensor self_cast;
    std::tie(mask_cast, self_cast) = expand_outplace_npu(mask, self);
    auto output_size = {mask_cast.numel()};
    return output_size;
}

at::Tensor& masked_select_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& mask)
{
    at::Tensor mask_bool = mask.dtype() == at::kBool ? mask : at_npu::native::custom_ops::npu_dtype_cast(mask, at::kBool);
    c10::SmallVector<int64_t, N> output_sync_idx = {0};
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("MaskedSelect")
        .Input(self)
        .Input(mask_bool)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& masked_select_out(const at::Tensor& self, const at::Tensor& mask, at::Tensor& out)
{
    at::Tensor masked_cast = mask.clone();
    auto output_size = masked_select_npu_output_size(self, masked_cast);
    npu_preparation::CheckOut(
        {self, masked_cast},
        out,
        ACL_FORMAT_ND,
        self.scalar_type(),
        output_size);

    masked_select_out_npu_nocheck(out, self, masked_cast);
    return out;
}

at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask)
{
    auto output_size = masked_select_npu_output_size(self, mask);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    masked_select_out_npu_nocheck(result, self, mask);
    return result;
}
} // namespace acl_op
