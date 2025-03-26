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
#include "op_plugin/utils/custom_functions/opapi/ForeachConstants.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
const char ROUND_MODE_FLOOR = char(2);
const char ROUND_MODE_CEIL = char(3);
const char ROUND_MODE_ROUND = char(1);
const char ROUND_MODE_TRUNC = char(5);
const char ROUND_MODE_FRAC = char(7);

bool is_integral_tensor_list(at::TensorList self)
{
    auto scalarType = self[0].scalar_type();
    return (scalarType == at::ScalarType::Byte
        || scalarType == at::ScalarType::Char
        || scalarType == at::ScalarType::Short
        || scalarType == at::ScalarType::Int
        || scalarType == at::ScalarType::Long);
}

void exec_npu_cmd_v2_(at::TensorList self, const char roundMode)
{
    if (is_integral_tensor_list(self)) {
        return;
    }
    at::Tensor round_mode_scalar_tensor = at_npu::native::OpPreparation::copy_scalar_to_device(
        roundMode, at::ScalarType::Char, self[0].device());
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber, self, round_mode_scalar_tensor, self);
}

std::vector<at::Tensor> exec_npu_cmd_v2(at::TensorList self, const char roundMode)
{
    bool is_integral = is_integral_tensor_list(self);
    auto scalarType = self[0].scalar_type();
    // construct the output tensorlist of the NPU
    std::vector<at::Tensor> result;
    for (uint32_t i = 0; i < self.size(); i++) {
        at::Tensor tensor = self[i];
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            at_npu::native::OpPreparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalarType))
        );
        if (is_integral) {
            result[i] = tensor.clone();
        }
    }

    if (is_integral) {
        return result;
    }

    at::TensorList result_ = at::TensorList(result);

    at::Tensor round_mode_scalar_tensor = at_npu::native::OpPreparation::copy_scalar_to_device(
        roundMode, at::ScalarType::Char, self[0].device());
    // dispatch hostAPI
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber, self, round_mode_scalar_tensor, result_);
    return result;
}

void _split_and_exec_npu_cmd_round(at::TensorList &tensors1, const char roundMode, at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? SINGLE_FOREACH_OP_TENSOR_COUNT : DOUBLE_FOREACH_OP_TENSOR_COUNT;
    size_t loop_time = tensor_count / max_tensor_count;
    auto roundModeScalar = at::Scalar(roundMode);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachRoundOffNumberV2, tensors1, roundModeScalar, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachRoundOffNumberV2, temp_tensors1, roundModeScalar, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count != 0) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachRoundOffNumberV2, temp_tensors1, roundModeScalar, temp_result);
    }
}

void exec_npu_cmd_(at::TensorList self, const char roundMode)
{
    if (is_integral_tensor_list(self)) {
        return;
    }
    // dispatch hostAPI
    _split_and_exec_npu_cmd_round(self, roundMode, self, true);
}

std::vector<at::Tensor> exec_npu_cmd(at::TensorList self, const char roundMode)
{
    bool is_integral = is_integral_tensor_list(self);
    auto scalarType = self[0].scalar_type();

    // construct the output tensorlist of the NPU
    std::vector<at::Tensor> result(self.size());
    for (size_t i = 0; i < self.size(); i++) {
        at::Tensor tensor = self[i];
        auto output_size = op_infer::input_same_output_size(tensor);
        if (is_integral) {
            result[i] = tensor.clone();
        } else {
            result[i] = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalarType));
        }
    }

    if (is_integral) {
        return result;
    }

    at::TensorList result_ = at::TensorList(result);

    // dispatch hostAPI
    _split_and_exec_npu_cmd_round(self, roundMode, result_, false);
    return result;
}

bool if_use_slow_route(at::TensorList tensors, const bool isFrac)
{
    at::native::check_foreach_api_restrictions(tensors);
    return !at::native::can_use_fast_route(tensors) || (isFrac && at::native::has_integral_tensor(tensors, true));
}

bool if_use_slow_route(at::TensorList tensors)
{
    return if_use_slow_route(tensors, false);
}

void _foreach_floor_(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_floor_slow_(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_floor_slow_(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2_(self, ROUND_MODE_FLOOR));
    exec_npu_cmd_(self, ROUND_MODE_FLOOR);
}

std::vector<at::Tensor> _foreach_floor(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_floor_slow(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_floor_slow(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2(self, ROUND_MODE_FLOOR));
    return exec_npu_cmd(self, ROUND_MODE_FLOOR);
}

void _foreach_ceil_(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_ceil_slow_(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_ceil_slow_(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2_(self, ROUND_MODE_CEIL));
    exec_npu_cmd_(self, ROUND_MODE_CEIL);
}

std::vector<at::Tensor> _foreach_ceil(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_ceil_slow(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_ceil_slow(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2(self, ROUND_MODE_FLOOR));
    return exec_npu_cmd(self, ROUND_MODE_CEIL);
}

void _foreach_round_(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_round_slow_(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_round_slow_(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2_(self, ROUND_MODE_ROUND));
    exec_npu_cmd_(self, ROUND_MODE_ROUND);
}

std::vector<at::Tensor> _foreach_round(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_round_slow(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_round_slow(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2(self, ROUND_MODE_ROUND));
    return exec_npu_cmd(self, ROUND_MODE_ROUND);
}

void _foreach_trunc_(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_trunc_slow_(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_trunc_slow_(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2_(self, ROUND_MODE_TRUNC));
    exec_npu_cmd_(self, ROUND_MODE_TRUNC);
}

std::vector<at::Tensor> _foreach_trunc(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_trunc_slow(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_trunc_slow(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2(self, ROUND_MODE_TRUNC));
    return exec_npu_cmd(self, ROUND_MODE_TRUNC);
}

void _foreach_frac_(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_frac_slow_(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_frac_slow_(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2_(self, ROUND_MODE_FRAC));
    exec_npu_cmd_(self, ROUND_MODE_FRAC);
}

std::vector<at::Tensor> _foreach_frac(at::TensorList self)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_frac_slow(self);
    }

    if (if_use_slow_route(self)) {
        return at::native::foreach_tensor_frac_slow(self);
    }
    DO_COMPATIBILITY(aclnnForeachRoundOffNumberV2, exec_npu_cmd_v2(self, ROUND_MODE_FRAC));
    return exec_npu_cmd(self, ROUND_MODE_FRAC);
}
} // namespace op_api
