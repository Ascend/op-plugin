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

#include "acl/acl.h"
#include "aclrtlaunch_matmul_leakyrelu_custom.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/op_tiling.h"
#include "utils.h"


namespace ascendc_path {

at::Tensor getTiling(optiling::TCubeTiling *tilingData, size_t tilingFileSize)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    auto buffer = at::empty({tilingSize}, at::kByte);
    tilingData->SaveToBuffer(buffer.data_ptr<uint8_t>(), tilingSize);
    auto tilingTensor = CopyTensorHostToDevice(buffer);
    return tilingTensor;
}

at::Tensor run_matmul_leakyrelu_custom(const at::Tensor &a, const at::Tensor &b, const at::Tensor &bias)
{
    auto c =
        at::empty({a.sizes()[0], b.sizes()[1]}, at::TensorOptions().dtype(at::kFloat).device(a.options().device()));

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t user_workspace_size = 0;
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    size_t workspace_size = user_workspace_size + system_workspace_size;
    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(a.options().device()));

    size_t tilingFileSize = sizeof(TCubeTiling);
    optiling::TCubeTiling tilingData = MatmulLeakyreluGenerateTiling();
    at::Tensor tiling = getTiling(&tilingData, tilingFileSize);
    uint32_t blockDim = 1;
    EXEC_KERNEL_CMD(matmul_leakyrelu_custom, blockDim, a, b, bias, c, workspace_tensor, tiling);
    return c;
}
} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("my_matmul_leakyrelu(Tensor x, Tensor y, Tensor bias) -> Tensor");
}
}

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("my_matmul_leakyrelu", TORCH_FN(ascendc_path::run_matmul_leakyrelu_custom));
}
}
