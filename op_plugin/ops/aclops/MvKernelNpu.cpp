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

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &mv_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &vec)
{
    bool is_self_t = op_plugin::utils::is_transpose_last_two_dims(self);
    at::Tensor contiguous_self = is_self_t ? self : npu_utils::format_contiguous(self);
    at::Tensor vec_t = at::unsqueeze(vec, 1);

    at_npu::native::OpCommand cmd;
    cmd.Name("MatMul")
        .InputWithoutContiguous(contiguous_self)
        .Input(vec_t)
        .Attr("transpose_x1", is_self_t)
        .Attr("transpose_x2", false)
        .Output(result)
        .Run();

    at_npu::native::npu_fast_reshape_(result);
    return result;
}
} // namespace

at::Tensor &mv_out(const at::Tensor &self, const at::Tensor &vec, at::Tensor &result)
{
    TORCH_CHECK(self.dim() >= 1, "mv(): input tensor must has at least 1 dimension, but got ", self.dim(),
        " dimensions" + OPS_ERROR(ErrCode::PARAM));
    npu_preparation::CheckOut({self}, result, self, {self.size(0)});

    result.unsqueeze_(1);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        mv_out_npu_nocheck(contiguous_result, self, vec);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        mv_out_npu_nocheck(result, self, vec);
    }
    result.squeeze_(1);
    return result;
}

at::Tensor mv(const at::Tensor &self, const at::Tensor &vec)
{
    TORCH_CHECK(self.dim() >= 1, "mv(): input tensor must has at least 1 dimension, but got ", self.dim(),
        " dimensions" + OPS_ERROR(ErrCode::PARAM));
    at::Tensor result = npu_preparation::apply_tensor(self, {self.size(0), 1});
    mv_out_npu_nocheck(result, self, vec);
    result.squeeze_(1);
    return result;
}
} // namespace acl_op
