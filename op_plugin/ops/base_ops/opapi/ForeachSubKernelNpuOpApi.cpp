// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_sub(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar &alpha)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow(tensors1, tensors2, alpha);
    }
    // construct the output tensorlist of the NPU
    auto scalar_type = tensors1[0].scalar_type();
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : tensors1) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    // convert scalar to tensor in PTA for now，wait for ascendc aclnn framwork support scalar type
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(alpha, scalar_type);

    EXEC_NPU_CMD(aclnnForeachSubList, tensors1, tensors2, scalar_, result_);
    return result;
}

void _foreach_sub_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar &alpha)
{
    at::native::check_foreach_api_restrictions(tensors1, tensors2);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route({tensors1, tensors2}, alpha)) {
        return at::native::foreach_tensor_sub_list_kernel_slow_(tensors1, tensors2, alpha);
    }
    // convert scalar to tensor in PTA for now，wait for ascendc aclnn framwork support scalar type
    auto scalar_type = tensors1[0].scalar_type();
    at::Tensor scalar_ = npu_preparation::copy_scalar_to_device(alpha, scalar_type);

    EXEC_NPU_CMD(aclnnForeachSubList, tensors1, tensors2, scalar_, tensors1);
    return;
}

std::vector<at::Tensor> _foreach_sub(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow(tensors, scalars);
}

void _foreach_sub_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    // default slow path for now, wait for ascendc aclnn framwork support scalarlist type
    at::native::check_foreach_api_restrictions(tensors, scalars);
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(tensors, scalars);
}

std::vector<at::Tensor> _foreach_sub(at::TensorList self, const at::Scalar &scalar)
{
    // Fallback
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow(self, scalar);
    }

    // Type Check
    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32");
    }

    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(
            npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type);
    EXEC_NPU_CMD(aclnnForeachSubScalar, self, scalar_tensor, result_);

    return result;
}

} // namespace op_api
