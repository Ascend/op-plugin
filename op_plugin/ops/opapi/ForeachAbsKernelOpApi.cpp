// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include <ATen/native/ForeachUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

void _split_and_exec_npu_cmd_abs(const at::TensorList tensors1, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = 1;
    if (is_inplace) {
        max_tensor_count = 48;
    } else {
        max_tensor_count = 24;
    }
    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
            EXEC_NPU_CMD(aclnnForeachAbs, tensors1, result_list);
            return;
        }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachAbs, temp_tensors1, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachAbs, temp_tensors1, temp_result);
    }
}

void _foreach_abs_(const at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachAbs, at::native::foreach_tensor_abs_slow_(self));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_abs_slow_(self);
    }
    
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_abs_slow_(self);
    }

    // datatype check
    if (!op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::BASE_DTYPE,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_abs_slow_(self);
    }

    if (self.empty()) {
        return;
    }
    _split_and_exec_npu_cmd_abs(self, self, true);
}


std::vector<at::Tensor> _foreach_abs(const at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachAbs, at::native::foreach_tensor_abs_slow(self));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_abs_slow(self);
    }

    if (!op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
                                               op_plugin::utils::ForeachTensorDtypeSupport::BASE_DTYPE,
                                               op_plugin::utils::ForeachInputType::TYPE_TENSOR)) {
        return at::native::foreach_tensor_abs_slow(self);
    }

    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_abs_slow(self);
    }

    auto scalar_type = self[0].scalar_type();

    // construct output tensorlist
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_abs(self, result_, false);
    return result;
}
} // namespace at_npu
