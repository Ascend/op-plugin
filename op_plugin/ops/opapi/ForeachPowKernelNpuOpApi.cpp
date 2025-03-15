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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"


namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;
constexpr size_t MAX_TENSOR_COUNT = 256;

void _split_and_exec_npu_cmd_pow(at::TensorList& tensors1, at::TensorList& tensors2, at::TensorList& result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;

    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachPowList, tensors1, tensors2, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_tensors2(tensors2.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachPowList, temp_tensors1, temp_tensors2, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count > 0) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_tensors2(tensors2.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachPowList, temp_tensors1, temp_tensors2, temp_result);
    }
}

void _split_and_exec_npu_cmd_pow_scalarlist(at::TensorList &tensors1, at::ArrayRef<at::Scalar> scalars,
                                            at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 24 : 16;
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachPowScalarList, tensors1, scalars, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachPowScalarList, temp_tensors1, temp_scalars, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count > 0) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        at::ArrayRef<at::Scalar> temp_scalars(scalars.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachPowScalarList, temp_tensors1, temp_scalars, temp_result);
    }
}

std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::TensorList exponent)
{
    DO_COMPATIBILITY(aclnnForeachPowList, at::native::foreach_tensor_pow_list_kernel_slow(self, exponent));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_pow_list_kernel_slow(self, exponent);
    }

    at::native::check_foreach_api_restrictions(self, exponent);
    if (!at::native::can_use_fast_route(self, exponent, true)) {
        return at::native::foreach_tensor_pow_list_kernel_slow(self, exponent);
    }

    // construct the output tensorlist of the NPU
    auto scalar_type = self[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_pow(self, exponent, result_, false);

    return result;
}

void _foreach_pow_(at::TensorList self, at::TensorList exponent)
{
    DO_COMPATIBILITY(aclnnForeachPowList, at::native::foreach_tensor_pow_list_kernel_slow_(self, exponent));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_pow_list_kernel_slow_(self, exponent);
    }

    at::native::check_foreach_api_restrictions(self, exponent);
    if (!at::native::can_use_fast_route(self, exponent, true)) {
        return at::native::foreach_tensor_pow_list_kernel_slow_(self, exponent);
    }

    _split_and_exec_npu_cmd_pow(self, exponent, self, true);
    return;
}

std::vector<at::Tensor> _foreach_pow(at::TensorList self, at::ArrayRef<at::Scalar> exponent)
{
    DO_COMPATIBILITY(aclnnForeachPowScalarList, at::native::foreach_tensor_pow_scalarlist_kernel_slow(self, exponent));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_pow_scalarlist_kernel_slow(self, exponent);
    }

    at::native::check_foreach_api_restrictions(self, exponent);
    if (!at::native::can_use_fast_route(self, exponent, true)) {
        return at::native::foreach_tensor_pow_scalarlist_kernel_slow(self, exponent);
    }

    auto scalar_type = self[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);
    _split_and_exec_npu_cmd_pow_scalarlist(self, exponent, result_, false);
    return result;
}

void _foreach_pow_(at::TensorList self, at::ArrayRef<at::Scalar> exponent)
{
    DO_COMPATIBILITY(aclnnForeachPowScalarList, at::native::foreach_tensor_pow_scalarlist_kernel_slow_(self, exponent));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_pow_scalarlist_kernel_slow_(self, exponent);
    }

    at::native::check_foreach_api_restrictions(self, exponent);
    if (!at::native::can_use_fast_route(self, exponent, true)) {
        at::native::foreach_tensor_pow_scalarlist_kernel_slow_(self, exponent);
        return;
    }

    _split_and_exec_npu_cmd_pow_scalarlist(self, exponent, self, true);
}
#endif

}  // namespace op_api
