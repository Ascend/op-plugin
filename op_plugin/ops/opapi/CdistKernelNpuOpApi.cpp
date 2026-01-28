// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static at::Tensor &cdist_out_op_api(const at::Tensor &x1, const at::Tensor &x2, float p, c10::optional<int64_t> compute_mode, at::Tensor &out)
{
    int64_t compute_mode_value = compute_mode.has_value() ? compute_mode.value() : 0;
    EXEC_NPU_CMD(aclnnCdist, x1, x2, p, compute_mode_value, out);
    return out;
}

at::Tensor cdist(const at::Tensor &x1, const at::Tensor &x2, const double p, c10::optional<int64_t> compute_mode)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::cdist(x1, x2, p, compute_mode);
    } else {
        float p_float;
        if (std::isinf(p)) {
            p_float = -1;
        } else {
            TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu does not support float64"
                + OPS_ERROR(ErrCode::TYPE));
            p_float = static_cast<float>(p);
        }
        auto output_size = op_infer::cdist_npu_output_size(x1, x2);
        auto type = x1.scalar_type();
        at::Tensor out = at::empty(output_size, x1.options().dtype(type));
        cdist_out_op_api(x1, x2, p, compute_mode, out);
        return out;
    }
}
}