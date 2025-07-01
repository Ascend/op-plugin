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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha)
{
    TORCH_CHECK(!alpha.isBoolean() || dtype == at::kBool, "Boolean alpha only supported for Boolean results."
        + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(isFloatingType(dtype) || alpha.isIntegral(true),
        "For integral input tensors, argument alpha must not be a floating point number."
        + OPS_ERROR(ErrCode::TYPE));
}

at::Tensor add_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return is_self_wrapped ? other : self;
}

at::Tensor &adds_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar other,
                                 const at::Scalar alpha)
{
    alpha_check_npu(result.scalar_type(), alpha);
    float other_value = op_plugin::utils::get_scalar_float_value(other);
    float alpha_value = op_plugin::utils::get_scalar_float_value(alpha);
    float value = other_value * alpha_value;
    at_npu::native::OpCommand cmd;
    std::string real_type = "";
    if (self.scalar_type() == at::kBool) {
        auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
        if (unified_result.common_type == at::kBool) {
            unified_result.common_type = at::kByte;
            unified_result.result_type_defined = true;
            real_type = "uint8";
        }
        cmd.Expect(unified_result);
    }
    cmd.Name("Add")
        .Input(self)
        .Input(at::Scalar(value), self.scalar_type())
        .Output(result, "", c10::nullopt, real_type)
        .Run();

    return result;
}

at::Tensor &add_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    auto unified_result = npu_preparation::binary_op_check(result, self, other, true);
    if (npu_preparation::IsCPUScalar(other)) {
        adds_out_npu_nocheck(result, self, other.item(), alpha);
    } else if (npu_preparation::IsCPUScalar(self)) {
        adds_out_npu_nocheck(result, other, self.item(), alpha);
    } else {
        alpha_check_npu(result.scalar_type(), alpha);
        at_npu::native::OpCommand cmd;
        cmd.Expect(unified_result);

        if (op_plugin::utils::is_scalar_one(alpha)) {
            if (self.scalar_type() == at::kLong) {
                TORCH_NPU_WARN_ONCE(
                    "The oprator of add is executed, Currently High Accuracy but Low Performance OP with 64-bit has "
                    "been used, Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
            }

            std::string real_type = "";
            if (self.scalar_type() == at::kBool && other.scalar_type() == at::kBool) {
                unified_result.common_type = at::kByte;
                unified_result.result_type_defined = true;
                cmd.Expect(unified_result);
                real_type = "uint8";
            }
            cmd.Name("Add").Input(self).Input(other).Output(result, "", c10::nullopt, real_type).Run();
        } else {
            cmd.Name("AxpyV2").Input(self).Input(other).Input(alpha, self.scalar_type()).Output(result).Run();
        }
    }
    return result;
}

bool check_size(const at::Tensor &self, const at::Tensor &other)
{
    if (self.dim() != other.dim()) {
        return false;
    }
    for (size_t i = 0; i < static_cast<uint64_t>(self.dim()); i++) {
        if (self.size(i) != other.size(i)) {
            return false;
        }
    }
    return true;
}

at::Tensor stride_add_tensor_get(const at::Tensor &src)
{
    if (src.is_contiguous()) {
        return src;
    } else {
        auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
        at::Tensor src_new =
            npu_preparation::apply_tensor_with_format(src_desc.base_sizes_, src.options(), ACL_FORMAT_NC1HWC0);
        src_new.set_(src.storage(), src_new.storage_offset(), src_new.sizes(), src_new.strides());
        return src_new;
    }
}
} // namespace

at::Tensor add(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    alpha_check_npu(at::native::result_type(self, other), alpha);
    if ((!(self.is_contiguous() && other.is_contiguous())) &&
        (npu_utils::check_5d_5d_match(self) || npu_utils::check_5d_5d_match(other)) && check_size(self, other)) {
        int64_t c0_len = 16;
        at::Tensor self_use = stride_add_tensor_get(self);
        TORCH_CHECK(self.numel() > 0, "self must be non-empty tensor" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(other.numel() > 0, "other must be non-empty tensor" + OPS_ERROR(ErrCode::PARAM));
        at::Scalar self_c1_offset(self.storage_offset() / (self.size(2) * self.size(3) * c0_len));
        at::Tensor other_use = stride_add_tensor_get(other);
        at::Scalar other_c1_offset(other.storage_offset() / (other.size(2) * other.size(3) * c0_len));
        at::Scalar stride_len(self.size(1) / c0_len);
        at::Tensor result = acl_op::npu_stride_add(self_use, other_use, self_c1_offset, other_c1_offset, stride_len);
        return result;
    }
    at::Tensor output_tensor = add_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = (self.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, output_tensor.options().dtype(result_type), npu_preparation::get_tensor_npu_format(output_tensor));
    add_out_npu_nocheck(result, self_cp, other_cp, alpha);
    return result;
}

at::Tensor add(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    alpha_check_npu(at::native::result_type(self, other), alpha);
    at::Tensor result = npu_preparation::apply_tensor(self);
    adds_out_npu_nocheck(result, self, other, alpha);
    return result;
}

at::Tensor &add_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    at::ScalarType result_type = at::native::result_type(self, other);
    at::ScalarType self_type = self.scalar_type();
    TORCH_CHECK(canCast(result_type, self_type), "result type ", result_type,
        " can't be cast to the desired output type ", self_type, OPS_ERROR(ErrCode::TYPE));
    at::Tensor self_cp = (self_type != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;

    npu_preparation::CheckMemory({self_cp, other_cp}, {self_cp});
    if (!npu_utils::check_match(&self_cp)) {
        at::Tensor contiguous_self;
        if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
            contiguous_self = npu_utils::format_contiguous_add_copy_optimize(self_cp);
        } else {
            contiguous_self = npu_utils::format_contiguous(self_cp);
        }
        add_out_npu_nocheck(contiguous_self, contiguous_self, other_cp, alpha);
        npu_utils::format_fresh_view(self_cp, contiguous_self);
    } else {
        add_out_npu_nocheck(self_cp, self_cp, other_cp, alpha);
    }

    if (self_type == result_type) {
        self = self_cp;
    } else {
        self.copy_(self_cp);
    }
    return self;
}

at::Tensor &add_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self;
        if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
            contiguous_self = npu_utils::format_contiguous_add_copy_optimize(self);
        } else {
            contiguous_self = npu_utils::format_contiguous(self);
        }
        adds_out_npu_nocheck(contiguous_self, contiguous_self, other, alpha);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        adds_out_npu_nocheck(self, self, other, alpha);
    }
    return self;
}

at::Tensor &add_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &result)
{
    at::Tensor output_tensor = add_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = (self.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                             at_npu::native::custom_ops::npu_dtype_cast(self, result_type) :
                             self;
    at::Tensor other_cp = (other.scalar_type() != result_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                              at_npu::native::custom_ops::npu_dtype_cast(other, result_type) :
                              other;

    npu_preparation::CheckOut({self_cp, other_cp}, result, npu_preparation::get_tensor_npu_format(result), result_type,
                              output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result;
        if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
            contiguous_result = npu_utils::format_contiguous_add_copy_optimize(result);
        } else {
            contiguous_result = npu_utils::format_contiguous(result);
        }
        add_out_npu_nocheck(contiguous_result, self_cp, other_cp, alpha);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        add_out_npu_nocheck(result, self_cp, other_cp, alpha);
    }
    return result;
}
} // namespace acl_op
