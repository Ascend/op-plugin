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

at::Tensor& put_(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& source,
    bool accumulate) {
    TORCH_CHECK(index.numel() == source.numel(), "source should have the same number of elements as index"
        + OPS_ERROR(ErrCode::PARAM));
    if (source.numel() == 0) {
        return self;
    }
    npu_preparation::CheckMemory({self, index, source}, {self});

    at::Tensor selfFlatten = npu_utils::format_contiguous(self.reshape(-1));
    at::Tensor indexFlatten = index.reshape({-1, 1});
    at::Tensor sourceFlatten = source.reshape(-1);

    at_npu::native::OpCommand cmd;
    accumulate ? cmd.Name("ScatterNdAdd") : cmd.Name("ScatterNdUpdate");
    cmd.Input(selfFlatten)
        .Input(indexFlatten)
        .Input(sourceFlatten)
        .Output(selfFlatten)
        .Attr("use_locking", false)
        .Run();

    self.copy_(selfFlatten);
    return self;
}
}  // namespace acl_op
