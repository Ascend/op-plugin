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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _split_and_exec_npu_cmd_cosh(at::TensorList tensors1, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachCosh, tensors1, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachCosh, temp_tensors1, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count > 0) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachCosh, temp_tensors1, temp_result);
    }
}

void _foreach_cosh_(at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachCosh, at::native::foreach_tensor_cosh_slow_(self));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_cosh_slow_(self);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::BASE_DTYPE,
        op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_cosh_slow_(self);
    }

    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_cosh_slow_(self);
    }

    if (self.empty()) {
        return;
    }

    _split_and_exec_npu_cmd_cosh(self, self, true);
}

std::vector<at::Tensor> _foreach_cosh(const at::TensorList tensors)
{
    DO_COMPATIBILITY(aclnnForeachCosh, at::native::foreach_tensor_cosh_slow(tensors));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_cosh_slow(tensors);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(tensors[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::BASE_DTYPE,
        op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_cosh_slow(tensors);
    }

    at::native::check_foreach_api_restrictions(tensors);
    if (!at::native::can_use_fast_route(tensors) || at::native::has_integral_tensor(tensors, true)) {
        return at::native::foreach_tensor_cosh_slow(tensors);
    }

    auto type = tensors[0].scalar_type();

    // construct output tensorlist
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_cosh(tensors, result_, false);
    return result;
}
} // namespace at_npu
