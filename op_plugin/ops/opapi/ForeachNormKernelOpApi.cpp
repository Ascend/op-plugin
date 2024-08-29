// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

constexpr int FOREACH_NORM_MAX_TENSOR_COUNT = 24;

bool checkData(const at::TensorList self, const at::Scalar& scalar)
{
    double p = 0.0;
    if (scalar.isIntegral(false)) {
        p = scalar.to<int64_t>();
    } else if (scalar.isFloatingPoint()) {
        p = scalar.to<double>();
    } else {
        TORCH_CHECK(false, "foreach_tensor_norm_npu expects scalar to be integer or float", OPS_ERROR(ErrCode::TYPE));
    }
    const bool has_int_or_complex = std::any_of(self.begin(), self.end(), [](const auto &t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, true) || at::isComplexType(scalar_type);
    });
    return !at::native::can_use_fast_route(self) || has_int_or_complex ||
           !(p == static_cast<double>(1) || p == static_cast<double>(2));
}

void _split_and_exec_npu_cmd_norm(at::TensorList tensors, const at::Scalar& scalar, at::TensorList result_list)
{
    size_t tensor_count = tensors.size();
    size_t max_tensor_count = FOREACH_NORM_MAX_TENSOR_COUNT;   // tensorlist切分长度

    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachNorm, tensors, scalar, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachNorm, temp_tensors1, scalar, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachNorm, temp_tensors1, scalar, temp_result);
    }
}

#if VERSION_BETWEEN(V1R11, V2R3)
std::vector<at::Tensor> _foreach_norm(const at::TensorList self, const at::Scalar& scalar)
{
    at::native::check_foreach_api_restrictions(self);

    static const bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1;
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_norm_slow(self, scalar);
    }

    DO_COMPATIBILITY(aclnnForeachNorm, at::native::foreach_tensor_norm_slow(self, scalar));

    if (checkData(self, scalar)) {
        return at::native::foreach_tensor_norm_slow(self, scalar);
    }
    auto scalar_type = self[0].scalar_type();

    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        result.push_back(npu_preparation::apply_tensor_without_format(1, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_norm(self, scalar, result_);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R4, VERSION_NEWEST)
std::vector<at::Tensor> _foreach_norm(const at::TensorList self, const at::Scalar& scalar, at::optional<at::ScalarType> opt_dtype)
{
    at::native::check_foreach_api_restrictions(self);

    static const bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1;
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_norm_slow(self, scalar, opt_dtype);
    }

    DO_COMPATIBILITY(aclnnForeachNorm, at::native::foreach_tensor_norm_slow(self, scalar, opt_dtype));

    if (checkData(self, scalar)) {
        return at::native::foreach_tensor_norm_slow(self, scalar, opt_dtype);
    }
    auto scalar_type = self[0].scalar_type();
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : scalar_type;
    TORCH_CHECK(promoteTypes(scalar_type, dtype) == dtype, "foreach_norm", ": the dtype of the input ", "(",
                scalar_type, ") should be convertible ", "without narrowing to the specified dtype (", dtype, ")", OPS_ERROR(ErrCode::TYPE));

    std::vector<at::Tensor> inputs;
    if (scalar_type != dtype) {
        for (const at::Tensor& tensor :self) {
            inputs.push_back(acl_op::npu_dtype_cast(tensor, dtype));
        }
    }

    at::TensorList inputs_ = at::TensorList(inputs);

    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        result.push_back(npu_preparation::apply_tensor_without_format(1, tensor.options().dtype(dtype)));
    }
    at::TensorList result_ = at::TensorList(result);

    if (scalar_type != dtype) {
        _split_and_exec_npu_cmd_norm(inputs_, scalar, result_);
    } else {
        _split_and_exec_npu_cmd_norm(self, scalar, result_);
    }
    return result;
}
#endif
}
