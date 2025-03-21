// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
using calcu_op_util = at_npu::native::CalcuOpUtil;

at::Tensor& npu_reshape_out(
    const at::Tensor& self,
    at::IntArrayRef shape,
    bool can_refresh,
    at::Tensor& out)
{
    if (can_refresh) {
        at_npu::native::StorageDescHelper::SetDesc(
            out,
            op_infer::array_to_small_vector(out.sizes()),
            op_infer::array_to_small_vector(out.strides()));
    } else {
        at_npu::native::copy_d2d_by_memcpy(
            out,
            self,
            at_npu::native::NPUNativeFunctions::get_storage_size(out));
    }
    return out;
}

at::Tensor npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh)
{
    at::Tensor result = npu_preparation::apply_tensor(self, shape);
    acl_op::npu_reshape_out(self, shape, can_refresh, result);

    return result;
}
} // namespace acl_op
