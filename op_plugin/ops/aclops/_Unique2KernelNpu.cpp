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
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> _unique2_out_npu(
    at::Tensor& y,
    at::Tensor& y_inverse,
    at::Tensor& y_counts,
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts)
{
    c10::SmallVector<int64_t, N> output_sync_idx = {0, 1, 2};
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("UniqueWithCountsAndSorting")
        .Input(self)
        .Output(y)
        .Output(y_inverse)
        .Output(y_counts)
        .Attr("sorted", sorted)
        .Attr("return_inverse", return_inverse)
        .Attr("return_counts", return_counts)
        .Run();

    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(y, y_inverse, y_counts);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts)
{
    /*
    * 算子去重调用的std::unordered_set会根据hash函数打乱顺序，
    * fp16场景与基本数据类型的打乱方式不同，使得sorted=false时，fp16精度不达标.
    * 此外，算子去重时，fp16存在数据精度损失，因此这里将fp16强转fp32处理.
    */
    const at::Tensor self_cast = self.scalar_type() == at::kHalf ?
        at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat) : self;
    if (self_cast.numel() == 0) {
        at::Tensor result = npu_preparation::apply_tensor(self_cast, {0});
        at::Tensor y_inverse = npu_preparation::apply_tensor({0}, self_cast.options().dtype(at::kLong), self_cast);
        at::Tensor y_counts = npu_preparation::apply_tensor({0}, self_cast.options().dtype(at::kLong), self_cast);
        return std::tie(result, y_inverse, y_counts);
    }
    at::Tensor y = npu_preparation::apply_tensor(self_cast, self_cast.numel());
    at::Tensor y_inverse = !(return_counts || return_inverse) ?
        npu_preparation::apply_tensor_with_format({1},
            self_cast.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format(self_cast.sizes(),
            self_cast.options().dtype(at::kLong), ACL_FORMAT_ND);
    at::Tensor y_counts = return_counts ?
        npu_preparation::apply_tensor_with_format(self_cast.numel(),
            self_cast.options().dtype(at::kLong), ACL_FORMAT_ND) :
        npu_preparation::apply_tensor_with_format({1},
            self_cast.options().dtype(at::kLong), ACL_FORMAT_ND);

    _unique2_out_npu(y, y_inverse, y_counts, self_cast, sorted, return_inverse, return_counts);
    if (self.scalar_type() == at::kHalf) {
        y = at_npu::native::custom_ops::npu_dtype_cast(y, at::kHalf);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, y_inverse, y_counts);
}
}  // namespace acl_op
