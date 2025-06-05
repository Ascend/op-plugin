// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool _logit_backward_fallback_condition()
{
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnLogitGrad");
    static const bool is_support_soc = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                        c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                        (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_aclnn_kernel_available || !is_support_soc) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::logit_backward and aten::logit_backward.out is currently "
            "not supported on the NPU backend. Now this operator will fallback to run on the CPU "
            "and may have performance implications.");
        return true;
    }
    return false;
}

at::Tensor &logit_backward_out(const at::Tensor &grad_output, const at::Tensor &self, c10::optional<double> eps, at::Tensor &grad_input)
{
    if (_logit_backward_fallback_condition()) {
        at::Tensor grad_output_cpu = grad_output.cpu();
        at::Tensor self_cpu = self.cpu();
        at::Tensor grad_input_cpu = grad_input.cpu();
        grad_input_cpu = at::logit_backward_outf(grad_output_cpu, self_cpu, eps, grad_input_cpu);
        grad_input.copy_(grad_input_cpu);
        return grad_input;
    }
    auto eps_value = eps.value_or(-1);
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogitGrad, self, grad_output, eps_value, grad_input);
    return grad_input;
}

at::Tensor logit_backward(const at::Tensor &grad_output, const at::Tensor &self, c10::optional<double> eps)
{
    if (_logit_backward_fallback_condition()) {
        at::Tensor grad_output_cpu = grad_output.cpu();
        at::Tensor self_cpu = self.cpu();
        at::Tensor out_cpu = at::logit_backward(grad_output_cpu, self_cpu, eps);
        return out_cpu.to(grad_output.device());
    }
    auto eps_value = eps.value_or(-1);
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                         grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogitGrad, self, grad_output, eps_value, grad_input);
    return grad_input;
}

}
