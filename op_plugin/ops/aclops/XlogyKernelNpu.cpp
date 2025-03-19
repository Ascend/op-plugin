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
at::Tensor &xlogy_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Xlogy").Input(self).Input(other).Output(result).Run();
    return result;
}

at::Tensor &xlogy_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Xlogy").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &xlogy_out_npu_nocheck(at::Tensor &result, const at::Scalar &self, const at::Tensor &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Xlogy").Input(self, other.scalar_type()).Input(other).Output(result).Run();
    return result;
}
} // namespace

at::Tensor &xlogy_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    at::Tensor format_cast_of_self = npu_preparation::CastBackToOriFormat(self);
    at::Tensor format_cast_of_other = npu_preparation::CastBackToOriFormat(other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self, other}, out, npu_preparation::get_tensor_npu_format(format_cast_of_self),
                              out.scalar_type(), output_size);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        xlogy_out_npu_nocheck(contiguous_result, format_cast_of_self, format_cast_of_other);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        xlogy_out_npu_nocheck(out, format_cast_of_self, format_cast_of_other);
    }
    return out;
}

at::Tensor &xlogy_out(const at::Tensor &self, const at::Scalar &other, at::Tensor &out)
{
    npu_preparation::CheckOut({self}, out, self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        xlogy_out_npu_nocheck(contiguous_result, self, other);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        xlogy_out_npu_nocheck(out, self, other);
    }
    return out;
}

at::Tensor &xlogy_out(const at::Scalar &self, const at::Tensor &other, at::Tensor &out)
{
    npu_preparation::CheckOut({other}, out, npu_preparation::get_tensor_npu_format(other), other.scalar_type(),
                              other.sizes());
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        xlogy_out_npu_nocheck(contiguous_result, self, other);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        xlogy_out_npu_nocheck(out, self, other);
    }
    return out;
}

at::Tensor xlogy(const at::Tensor &self, const at::Tensor &other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor output_tensor = is_self_wrapped ? other : self;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(output_tensor, output_size);
    xlogy_out_npu_nocheck(result, self, other);
    return result;
}

at::Tensor xlogy(const at::Tensor &self, const at::Scalar &other)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    xlogy_out_npu_nocheck(result, self, other);
    return result;
}

at::Tensor xlogy(const at::Scalar &self, const at::Tensor &other)
{
    at::Tensor result = npu_preparation::apply_tensor(other);
    xlogy_out_npu_nocheck(result, self, other);
    return result;
}

at::Tensor &xlogy_(at::Tensor &self, const at::Tensor &other)
{
    npu_preparation::CheckMemory({self, other}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        at::Tensor result = xlogy_out_npu_nocheck(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        xlogy_out_npu_nocheck(self, self, other);
    }
    return self;
}

at::Tensor &xlogy_(at::Tensor &self, const at::Scalar &other)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        xlogy_out_npu_nocheck(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        xlogy_out_npu_nocheck(self, self, other);
    }
    return self;
}
} // namespace acl_op
