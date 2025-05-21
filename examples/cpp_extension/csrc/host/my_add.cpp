// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.h"
#include "aclrtlaunch_add_custom.h"

namespace ascendc_path {

at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}
} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("my_add", TORCH_FN(ascendc_path::run_add_custom));
}
}
