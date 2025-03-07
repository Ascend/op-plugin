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

std::tuple<at::Tensor, at::Tensor> batch_norm_stats(const at::Tensor& self, double eps)
{
    TORCH_CHECK(
        self.ndimension() >= 2,
        "Expected 2D+ Tensor, but got tensor with ",
        self.ndimension(),
        " Dimension" + OPS_ERROR(ErrCode::PARAM));
    auto output_size = {self.size(1)};
    at::Tensor mean = npu_preparation::apply_tensor(output_size, self.options().dtype(at::kFloat), self);
    at::Tensor invstd = npu_preparation::apply_tensor(output_size, self.options().dtype(at::kFloat), self);

    c10::SmallVector<int64_t, N> dim;
    int dimN = self.ndimension();
    for (int i = 0; i < dimN; i++) {
        if (i == 1) {
            continue;
        }
        dim.emplace_back(i);
    }

    at::Tensor self_copy = self;
    if (self.scalar_type() != at::kFloat) {
        self_copy = at_npu::native::custom_ops::npu_dtype_cast(self_copy, at::kFloat);
    }

    at_npu::native::OpCommand cmd_mean;
    cmd_mean.Name("ReduceMean")
        .Input(self_copy)
        .Input(dim, at::kInt)
        .Output(mean)
        .Attr("keep_dims", (bool) false)
        .Run();

    at::Tensor mean_copy = mean;
    if (mean.dim() != 0) {
        auto dim_vector = op_infer::array_to_small_vector(dim);
        for (uint64_t i = 0; i < dim_vector.size(); i++) {
            mean_copy = mean_copy.unsqueeze(dim_vector[i]);
        }
    }
    mean_copy = mean_copy.expand(self.sizes());
    at_npu::native::OpCommand cmd_invstd;
    cmd_invstd.Name("ReduceStdWithMean")
        .Input(self_copy)
        .Input(mean_copy)
        .Output(invstd)
        .Attr("dim", dim)
        .Attr("unbiased", false)
        .Attr("keepdim", false)
        .Attr("invert", true)
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    return std::tie(mean, invstd);
}
} // namespace acl_op
