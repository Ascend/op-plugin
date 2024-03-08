#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void _foreach_sub_(const at::TensorList self, const at::Scalar &scalar)
{
    at::native::check_foreach_api_restrictions(self);
    if (!at_npu::native::env::CheckJitDisable() ||
        !at::native::can_use_fast_route(self, scalar, false)) {
        return at::native::foreach_tensor_sub_scalar_kernel_slow_(self, scalar);
    }

    auto scalar_type = self[0].scalar_type();
    if (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::Float &&
        scalar_type != at::ScalarType::Int) {
        TORCH_CHECK(false, "input must be half, float or int32", OPS_ERROR(ErrCode::TYPE));
    }
    at::Tensor scalar_tensor = npu_preparation::copy_scalar_to_device(scalar, scalar_type, self[0].device());
    EXEC_NPU_CMD(aclnnForeachSubScalar, self, scalar_tensor, self);
}

} // namespace op_api