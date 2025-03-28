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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &im2col_out(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                       at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnIm2col, acl_op::im2col_out(self, kernel_size, dilation, padding, stride, out));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::im2col_out(self, kernel_size, dilation, padding, stride, out);
    }
    auto output_size = op_infer::image_to_col_npu_output_size(self, kernel_size, stride, dilation, padding);
    npu_preparation::check_tensor({self}, out, out.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnIm2col, self, kernel_size, dilation, padding, stride, out);
    return out;
}

at::Tensor im2col(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                  at::IntArrayRef padding, at::IntArrayRef stride)
{
    DO_COMPATIBILITY(aclnnIm2col, acl_op::im2col(self, kernel_size, dilation, padding, stride));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::im2col(self, kernel_size, dilation, padding, stride);
    }
    auto output_size = op_infer::image_to_col_npu_output_size(self, kernel_size, stride, dilation, padding);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnIm2col, self, kernel_size, dilation, padding, stride, result);
    return result;
}

}
