// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool _logit_fallback_condition()
{
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnLogit");
    static const bool is_support_soc = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                       c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                       (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_aclnn_kernel_available || !is_support_soc) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::logit, aten::logit_ and aten::logit.out is currently "
            "not supported on the NPU backend. Now this operator will fallback to run on the CPU "
            "and may have performance implications.");
        return true;
    }
    return false;
}

at::Tensor logit(const at::Tensor &self, c10::optional<double> eps)
{
    if (_logit_fallback_condition()) {
        at::Tensor self_cpu = self.cpu();
        at::Tensor out_cpu = at::native::logit(self_cpu, eps);
        return out_cpu.to(self.device());
    }
    auto eps_value = eps.value_or(-1);
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0, self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogit, self, eps_value, out);
    return out;
}

at::Tensor &logit_(at::Tensor &self, c10::optional<double> eps)
{
    return at::native::logit_(self, eps);
}

at::Tensor &logit_out(const at::Tensor &self, c10::optional<double> eps, at::Tensor &out)
{
    if (_logit_fallback_condition()) {
        at::Tensor self_cpu = self.cpu();
        at::Tensor out_cpu = out.cpu();
        out_cpu = at::native::logit_out(self_cpu, eps, out_cpu);
        out.copy_(out_cpu);
        return out;
    }
    auto eps_value = eps.value_or(-1);
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogit, self, eps_value, out);
    return out;
}

}
