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
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "op_plugin/utils/OpAdapter.h"


namespace acl_op {

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor _weight_norm(
    const at::Tensor& v_in,
    const at::Tensor& g_in,
    int64_t dim)
{
    TORCH_CHECK(
        v_in.device() == g_in.device(),
        "weight_norm: expected v_in and g_in to be on the same device, but v_in is "
        "on ", v_in.device(), " and g_in is on ", g_in.device(), OPS_ERROR(ErrCode::PARAM));
    auto v = v_in.contiguous();
    auto g = g_in.contiguous();
    return v * g.div(at::norm_except_dim(v, 2, dim));
}
#endif

} // namespace acl_op
