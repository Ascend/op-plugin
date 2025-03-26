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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& embedding_renorm_gather2d_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GatherV2D")
        .Input(self)
        .Input(indices)
        .Output(result)
        .Attr("axis", static_cast<int64_t>(0))
        .Run();
    return result;
}

at::Tensor& embedding_renorm_execute_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    double max_norm,
    double norm_type)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Renorm")
        .Input(self)
        .Output(result)
        .Attr("p",  static_cast<float>(norm_type))
        .Attr("dim", static_cast<int64_t>(0))
        .Attr("maxnorm", static_cast<float>(max_norm))
        .Run();
    return result;
}

at::Tensor& embedding_renorm_scatter_update_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterUpdate")
        .Input(self)
        .Input(indices)
        .Input(update)
        .Output(result)
        .Attr("use_locking", false)
        .Run();
    return result;
}

at::Tensor& embedding_renorm_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& indices,
    double max_norm,
    double norm_type)
{
    at::SmallVector<int64_t, SIZE> mid_size = {indices.size(0), self.size(1)};
    at::Tensor mid_input = npu_preparation::apply_tensor(self, mid_size);
    at::Tensor mid_output = npu_preparation::apply_tensor(self, mid_size);

    embedding_renorm_gather2d_nocheck(mid_input, self, indices);
    embedding_renorm_execute_nocheck(mid_output, mid_input, max_norm, norm_type);

    auto num_indices = indices.numel();
    at::Tensor input_indices;

    if (num_indices - 1 == 0) {
        input_indices = at_npu::native::custom_ops::npu_dtype_cast(at::zeros({1}, self.options()), at::kLong);
    } else {
        input_indices = at_npu::native::custom_ops::npu_dtype_cast(at::range(0, num_indices - 1, self.options()), at::kLong);
    }

    auto num_mid_output = mid_output.numel();
    mid_output.resize_(num_mid_output);
    at::Tensor scalar_out = npu_preparation::apply_tensor(self, {num_indices, 1});
    embedding_renorm_gather2d_nocheck(scalar_out, mid_output, input_indices);

    at::Tensor out_res = mid_input * scalar_out;
    embedding_renorm_scatter_update_nocheck(result, self, indices, out_res);
    return result;
}
} // namespace

at::Tensor& embedding_renorm_(
    at::Tensor& self,
    const at::Tensor& indices,
    double max_norm,
    double norm_type)
{
    auto self_arg = at::TensorArg(self, "self", 1);
    auto indices_arg = at::TensorArg(indices, "indices", 2);
    at::checkDim("embedding_renorm_", self_arg, 2);
    at::checkScalarType("embedding_renorm_", indices_arg, at::kLong);

    auto num_indices = indices.numel();
    TORCH_CHECK(num_indices >= 1, "indices.numel() must be greater than or equal to 1, but got ", num_indices,
                OPS_ERROR(ErrCode::PARAM));
    at::native::resize_(indices, num_indices);

    npu_preparation::CheckMemory({self, indices}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        embedding_renorm_out_npu_nocheck(contiguous_self, contiguous_self, indices, max_norm, norm_type);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        embedding_renorm_out_npu_nocheck(self, self, indices, max_norm, norm_type);
    }
    return self;
}
} // namespace acl_op
