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
    static const bool is_adaptive_max_pool_3d_available = check_aclnn_kernel_available("aclnnAdaptiveMaxPool3d");
    if (!is_adaptive_max_pool_3d_available || dtype == at::kDouble) {
        return false;
    }
    return true;
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool3d_out(const at::Tensor& self, at::IntArrayRef output_size, at::Tensor& out, at::Tensor& indices)
{
    if (!is_npu_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("adaptive_max_pool3d.out is not supported by NPU currently. Now this kernel is running on CPU.");
        auto out_cpu = out.cpu();
        auto indices_cpu = indices.cpu();
        auto cpuout = at::adaptive_max_pool3d_out(out_cpu, indices_cpu, self.cpu(), output_size);
        out.copy_(std::get<0>(cpuout));
        indices.copy_(std::get<1>(cpuout));
        return std::tuple<at::Tensor&, at::Tensor&>(out, indices);
    }

    auto out_size = op_infer::max_pool3d_output_size(self, output_size);
    npu_preparation::check_tensor({self}, out, self.scalar_type(), out_size);
    npu_preparation::check_tensor({self}, indices, at::kInt, out_size);

    EXEC_NPU_CMD(aclnnAdaptiveMaxPool3d, self, output_size, out, indices);
    return std::tuple<at::Tensor&, at::Tensor&>(out, indices);
}

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d(const at::Tensor& self, at::IntArrayRef output_size)
{
    if (!is_npu_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("adaptive_max_pool3d is not supported by NPU currently. Now this kernel is running on CPU.");
        auto result = at::adaptive_max_pool3d(self.cpu(), output_size);
        auto out_npu = std::get<0>(result).to(self.device());
        auto indices_npu = std::get<1>(result).to(self.device());
        return std::tuple<at::Tensor, at::Tensor>(out_npu, indices_npu);
    }

    auto out_size = op_infer::max_pool3d_output_size(self, output_size);
    at::Tensor out = npu_preparation::apply_tensor_without_format(out_size, self.options());
    at::Tensor indices = npu_preparation::apply_tensor_without_format(out_size, self.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnAdaptiveMaxPool3d, self, output_size, out, indices);
    return std::tuple<at::Tensor, at::Tensor>(out, indices);
}

}
