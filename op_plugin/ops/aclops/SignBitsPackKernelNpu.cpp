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

at::Tensor npu_sign_bits_pack(const at::Tensor& self, int64_t size) {
    TORCH_CHECK(self.dim() == 1, "input must be one-dimensional" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Float,
        "all only supports torch.float16 and torch.float32 dtypes" + OPS_ERROR(ErrCode::TYPE));
    auto ysize = (self.numel() + 7) / 8;
    TORCH_CHECK(size != 0 && ysize % size == 0, "all must be divisible by size" + OPS_ERROR(ErrCode::PARAM));
    at::Tensor result = npu_preparation::apply_tensor({size, ysize / size}, self.options().dtype(at::kByte), self);

    at_npu::native::OpCommand cmd;
    cmd.Name("SignBitsPack")
        .Input(self)
        .Output(result)
        .Attr("size", size)
        .Run();
    return result;
}

} // namespace acl_op
