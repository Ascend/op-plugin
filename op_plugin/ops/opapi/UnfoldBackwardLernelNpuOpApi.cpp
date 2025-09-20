// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool _unfold_backward_fallback_condition()
{
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnUnfoldGrad");
    static const bool is_support_soc = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                       c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                       (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910_9391);
    static const bool is_cann_ready = op_plugin::utils::is_gte_cann_version_830rc1();
    if (!is_aclnn_kernel_available || !is_support_soc || !is_cann_ready) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::unfold_backward is currently "
            "not supported on the NPU backend. Now this operator will fallback to run on the CPU "
            "and may have performance implications.");
        return true;
    }
    return false;
}

at::Tensor unfold_backward(const at::Tensor & grad_in, at::IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step)
{
    if (_unfold_backward_fallback_condition()) {
        at::Tensor grad_in_cpu = grad_in.cpu();
        at::Tensor result = at::unfold_backward(grad_in_cpu, input_sizes, dim, size, step);
        return result.to(grad_in.device());
    }

    auto output_size = op_infer::array_to_small_vector(input_sizes);
    auto output_dtype = grad_in.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, grad_in.options().dtype(output_dtype));
    EXEC_NPU_CMD(aclnnUnfoldGrad, grad_in, input_sizes, dim, size, step, out);
    return out;
}

}  // namespace op_api
