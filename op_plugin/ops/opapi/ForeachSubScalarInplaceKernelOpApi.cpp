#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/scalar_op_api.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _split_and_exec_npu_cmd_sub_scalar(at::TensorList tensors1, const at::Scalar &scalar, at::TensorList &result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? 48 : 24;

    size_t loop_time = tensor_count / max_tensor_count;
    at::Scalar scalar_ = op_api::adaptToDouble(scalar, tensors1);

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachSubScalar, tensors1, scalar_, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachSubScalar, temp_tensors1, scalar_, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachSubScalar, temp_tensors1, scalar_, temp_result);
    }
}

std::vector<at::Tensor> _foreach_sub(at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow(self, scalar);
    }

    // Fallback
    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow(self, scalar);
    }

    // Type Check
    auto scalar_type = self[0].scalar_type();
    TORCH_CHECK(scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float ||
                scalar_type == at::ScalarType::Int || scalar_type == at::ScalarType::BFloat16,
                "input must be half, float, int32 or bfloat16");
    
    std::vector<at::Tensor> result(self.size());
    auto iterRes = result.data();
    int i = 0;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        iterRes[i++] = npu_preparation::apply_tensor_without_format(output_size, tensor.options().dtype(scalar_type));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_sub_scalar(self, scalar, result_, false);

    return result;
}

void _foreach_sub_(at::TensorList self, const at::Scalar& scalar)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow_(self, scalar);
    }

    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();

    TORCH_CHECK(scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float ||
                scalar_type == at::ScalarType::Int || scalar_type == at::ScalarType::BFloat16,
                "input must be half, float, int32 or bfloat16");
    _split_and_exec_npu_cmd_sub_scalar(self, scalar, self, true);
}
}
