// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor& gelu_out(const at::Tensor & self, c10::string_view approximate, at::Tensor& out)
{
    static auto gelu_sc = at_npu::native::env::CheckCompatibleImpl();
    if (!gelu_sc) {
        DO_COMPATIBILITY(aclnnGelu, acl_op::gelu_out(self, approximate, out));
        EXEC_NPU_CMD(aclnnGelu, self, out);
    } else {
        auto approximate_mode = op_infer::npu_gelu_approximate_mode(approximate);
        EXEC_NPU_CMD(aclnnGeluV2, self, approximate_mode, out);
    }
    return out;
}
}