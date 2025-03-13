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
at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other)
{
    auto unified_result = npu_preparation::comparison_op_check(result, self, other, true);
    if (self.scalar_type() == at::kLong) {
        TORCH_NPU_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP "
            "with 64-bit has been used, Please Do Some Cast at Python Functions with 32-bit for "
            "Better Performance!");
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("NotEqual")
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();

    return result;
}

at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other)
{
    if (self.scalar_type() == at::kLong) {
        TORCH_NPU_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP "
            "with 64-bit has been used, Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("NotEqual")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor& ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::ne_out(self, other.item(), out);
    } else if (npu_preparation::IsCPUScalar(self)) {
        return acl_op::ne_out(other, self.item(), out);
    } else {
        auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
        npu_preparation::CheckOut({self, other}, out, out, output_size);

        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device(),
            OPS_ERROR(ErrCode::PARAM));

        at::ScalarType calculate_type = at::native::result_type(self, other);
        auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
        auto other_cast = op_plugin::utils::get_cast_input(other, calculate_type);

        auto out_type = out.scalar_type();
        at::Tensor out_cast = (out_type != at::kBool) ?
            at_npu::native::custom_ops::npu_dtype_cast(out, at::kBool) : out;
        if (!npu_utils::check_match(&out_cast)) {
            at::Tensor contiguous_out = npu_utils::format_contiguous(out_cast);
            ne_out_npu_nocheck(contiguous_out, self_cast, other_cast);
            npu_utils::format_fresh_view(out_cast, contiguous_out);
        } else {
            ne_out_npu_nocheck(out_cast, self_cast, other_cast);
        }

        if (out_type != at::kBool) {
            out_cast = at_npu::native::custom_ops::npu_dtype_cast(out_cast, out_type);
            out.copy_(out_cast);
        }
        return out;
    }
}

at::Tensor& ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out)
{
    at::ScalarType calculate_type = at::native::result_type(self, other);
    auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
    npu_preparation::CheckOut({self}, out, out, self.sizes());

    auto out_type = out.scalar_type();
    at::Tensor out_cast = (out_type != at::kBool) ?
        at_npu::native::custom_ops::npu_dtype_cast(out, at::kBool) : out;
    if (!npu_utils::check_match(&out_cast)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out_cast);
        ne_out_npu_nocheck(contiguous_out, self_cast, other);
        npu_utils::format_fresh_view(out_cast, contiguous_out);
    } else {
        ne_out_npu_nocheck(out_cast, self_cast, other);
    }

    if (out_type != at::kBool) {
        out_cast = at_npu::native::custom_ops::npu_dtype_cast(out_cast, out_type);
        out.copy_(out_cast);
    }
    return out;
}

at::Tensor ne(const at::Tensor& self, const at::Tensor& other)
{
    if (npu_preparation::IsCPUScalar(other)) {
        return acl_op::ne(self, other.item());
    } else if (npu_preparation::IsCPUScalar(self)) {
        return acl_op::ne(other, self.item());
    } else {
        TORCH_CHECK(self.device() == other.device(),
            "Expected all tensors to be on the same device, but found at least two devices, ",
            self.device(), " and ", other.device(),
            OPS_ERROR(ErrCode::PARAM));

        at::ScalarType calculate_type = at::native::result_type(self, other);
        auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
        auto other_cast = op_plugin::utils::get_cast_input(other, calculate_type);

        auto output_size = op_infer::broadcast_ops_npu_output_size(self_cast, other_cast);
        at::Tensor result = npu_preparation::apply_tensor(output_size,
                                                          self_cast.options().dtype(at::kBool), self_cast);
        ne_out_npu_nocheck(result, self_cast, other_cast);
        return result;
    }
}

at::Tensor ne(const at::Tensor& self, const at::Scalar& other)
{
    at::ScalarType calculate_type = at::native::result_type(self, other);
    auto self_cast = op_plugin::utils::get_cast_input(self, calculate_type);
    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(at::kBool));
    ne_out_npu_nocheck(result, self_cast, other);
    return result;
}

at::Tensor& ne_(at::Tensor& self, const at::Tensor& other)
{
    return acl_op::ne_out(self, other, self);
}

at::Tensor& ne_(at::Tensor& self, const at::Scalar& other)
{
    return acl_op::ne_out(self, other, self);
}
} // namespace acl_op
