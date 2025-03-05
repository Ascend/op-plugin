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

namespace {
at::Tensor& repeat_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef repeats)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Tile")
        .Input(self)
        .Input(repeats)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor repeat(const at::Tensor& self, at::IntArrayRef repeats)
{
    TORCH_CHECK(repeats.size() >= self.ndimension(),
                "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
                + OPS_ERROR(ErrCode::PARAM));
    at::Tensor self_cp = self;
    if (static_cast<int>(repeats.size()) > self_cp.ndimension()) {
        int diff = static_cast<int>(repeats.size()) - self_cp.ndimension();
        for (int i = 0; i < diff; i++) {
            self_cp = at::unsqueeze(self_cp, 0);
        }
    }

    auto output_size = op_infer::repeat_npu_output_size(self_cp, repeats);
    at::Tensor result = npu_preparation::apply_tensor(self_cp, output_size);

    repeat_out_npu_nocheck(result, self_cp, repeats);
    return result;
}
} // namespace acl_op
