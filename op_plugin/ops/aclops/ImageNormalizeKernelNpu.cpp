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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& image_normalize_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype)
{
    TORCH_CHECK(mean.has_value() && variance.has_value(),
                "[mean] and [variance] should be mandatory"
                + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(dtype == 0 || dtype == 1,
                "output data type should be float16 or float32"
                + OPS_ERROR(ErrCode::TYPE));
    std::vector<int64_t> para_shape = {1, 3, 1, 1};
    at_npu::native::OpCommand cmd;
    cmd.Name("NormalizeV2")
        .Input(self)
        .Input(mean.value(), para_shape, at::kFloat)
        .Input(variance.value(), para_shape, at::kFloat)
        .Output(result)
        .Attr("dtype", dtype)
        .Run();

    return result;
}
} // namespace

at::Tensor image_normalize(
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype)
{
    at::Tensor result;
    if (dtype == 0) {
        result = npu_preparation::apply_tensor(self, self.options().dtype(at::kFloat));
    } else {
        result = npu_preparation::apply_tensor(self, self.options().dtype(at::kHalf));
    }
    image_normalize_out_nocheck(result, self, mean, variance, dtype);
    return result;
}

at::Tensor& image_normalize_(
    at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    int64_t dtype)
{
    TORCH_CHECK(self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf,
                "inplace image normalize can only support float16 or float32"
                + OPS_ERROR(ErrCode::TYPE));
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        image_normalize_out_nocheck(contiguous_self, contiguous_self, mean, variance, dtype);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        image_normalize_out_nocheck(self, self, mean, variance, dtype);
    }
    return self;
}
} // namespace acl_op
