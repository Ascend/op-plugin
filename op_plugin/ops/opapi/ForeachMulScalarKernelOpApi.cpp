// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_mul_v1(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32", OPS_ERROR(ErrCode::TYPE));
    }
    
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, self[0].scalar_type(), self[0].device());

    EXEC_NPU_CMD(aclnnForeachMulScalar, self, scalar_tensor, result_);

    return result;
}

void _foreach_mul_v1_(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32", OPS_ERROR(ErrCode::TYPE));
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type, self[0].device());
    EXEC_NPU_CMD(aclnnForeachMulScalar, self, scalar_tensor, self);
}

void _split_and_exec_npu_cmd_mul(const at::TensorList tensors1, const at::Scalar &scalar, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;

    at::Scalar scalar_ = op_api::adaptToDouble(scalar, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachMulScalarV2, tensors1, scalar_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachMulScalarV2, temp_tensors1, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachMulScalarV2, temp_tensors1, scalar_, temp_result);
    }
}

std::vector<at::Tensor> _foreach_mul(const at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow(self, scalar);
    }
    DO_COMPATIBILITY(aclnnForeachMulScalarV2, _foreach_mul_v1(self, scalar));
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_mul(self, scalar, result_, false);

    return result;
}

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
    }
    DO_COMPATIBILITY(aclnnForeachMulScalarV2, _foreach_mul_v1_(self, scalar));
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_mul_scalar_kernel_slow_(self, scalar);
    }

    _split_and_exec_npu_cmd_mul(self, scalar, self, true);
}
}
