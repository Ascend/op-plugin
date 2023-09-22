#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_mul_(const at::TensorList self, const at::Scalar& scalar) {
    if (self.empty()) {
        return;
    }
    auto scalar_type = self[0].scalar_type();
    bool is_support = true;
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float && scalar_type != at::ScalarType::Int) {
        is_support = false;
        TORCH_CHECK(is_support, "input must be half, float or int32");
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, self[0].scalar_type());
    EXEC_NPU_CMD(aclnnForeachMulScalarInplace, self, scalar_tensor);
}
}