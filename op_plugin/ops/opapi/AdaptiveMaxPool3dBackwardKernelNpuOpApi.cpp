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
namespace {
bool is_npu_supported(at::ScalarType dtype)
{
    static const bool is_adaptive_max_pool_3d_backward_available = check_aclnn_kernel_available("aclnnAdaptiveMaxPool3dBackward");
    if (!is_adaptive_max_pool_3d_backward_available || dtype == at::kDouble) {
        return false;
    }
    return true;
}
} // namespace

at::Tensor& adaptive_max_pool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::Tensor& grad_input)
{
    if (!is_npu_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("adaptive_max_pool3d_backward.grad_input is not supported by NPU currently. Now this kernel is running on CPU.");
        auto grad_input_cpu = grad_input.cpu();
        auto cpuout = at::adaptive_max_pool3d_backward_out(grad_input_cpu, grad_output.cpu(), self.cpu(), indices.cpu());
        grad_input.copy_(cpuout);
        return grad_input;
    }

    npu_preparation::check_tensor({grad_output, self, indices}, grad_input, grad_output.scalar_type(), self.sizes());

    EXEC_NPU_CMD(aclnnAdaptiveMaxPool3dBackward, grad_output, self, indices, grad_input);
    return grad_input;
}

at::Tensor adaptive_max_pool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices)
{
    if (!is_npu_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("adaptive_max_pool3d_backward is not supported by NPU currently. Now this kernel is running on CPU.");
        auto grad_input_cpu = at::adaptive_max_pool3d_backward(grad_output.cpu(), self.cpu(), indices.cpu());
        auto grad_input_npu = grad_input_cpu.to(grad_output.device());
        return grad_input_npu;
    }

    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(self.sizes(), grad_output.options());

    EXEC_NPU_CMD(aclnnAdaptiveMaxPool3dBackward, grad_output, self, indices, grad_input);
    return grad_input;
}

}
