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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, SIZE> median_npu_output_size(const at::Tensor &self, int64_t dim, bool keepdim)
{
    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    at::IntArrayRef dims(dim);
    return op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
}
} // namespace

at::Tensor nanmedian(const at::Tensor &self)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [nanmedian] is not supported by NPU currently. Now this kernel is running on CPU.");
    at::Tensor self_cpu = self.to("cpu");
    auto result = at::native::nanmedian_cpu(self_cpu);
    at::Tensor output = result.to(self.device());
    return output;
}

std::tuple<at::Tensor, at::Tensor> nanmedian(const at::Tensor &self, int64_t dim, bool keepdim)
{
    TORCH_WARN_ONCE(
        "Warning: kernel [nanmedian.dim] is not supported by NPU currently. Now this kernel is running on CPU.");
    auto output_size = median_npu_output_size(self, dim, keepdim);
    at::Tensor values = npu_preparation::apply_tensor_with_format(output_size, self.options(),
                                                                  npu_preparation::get_tensor_npu_format(self));
    at::Tensor indices =
        npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kLong), ACL_FORMAT_NCHW);

    auto self_cpu = self.cpu();
    auto values_cpu = values.cpu();
    auto indices_cpu = indices.cpu();
    auto result = at::native::nanmedian_out_cpu(self_cpu, dim, keepdim, values_cpu, indices_cpu);
    at::Tensor values_out = values_cpu.to(self.device());
    at::Tensor indices_out = indices_cpu.to(self.device());
    return std::tuple<at::Tensor &, at::Tensor &>(values_out, indices_out);
}
} // namespace acl_op
