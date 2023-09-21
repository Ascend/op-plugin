#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_sub_(const at::TensorList self, const at::Scalar& scalar)
{
    if (self.empty()) {
        return;
    }

    auto iter = std::find_if(self.begin(), self.end(), [](const at::Tensor& tensor) {
        return tensor.numel() != 0;
    });
    if (iter == self.end()) {
        return;
    }

    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32");
    }
    at::Tensor scalar_tensor = at_npu::native::CalcuOpUtil::CopyScalarToDevice(scalar, self[0].scalar_type());
    EXEC_NPU_CMD(aclnnForeachSubScalarInplace, self, scalar_tensor);
}

}