#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_addcdiv(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars);
    // 暂不支持对scalarlist的处理，暂时走cuda原生路径
    return at::native::foreach_tensor_addcdiv_scalarlist_slow(input, tensors1, tensors2, scalars);
}

void _foreach_addcdiv_(const at::TensorList input,
    const at::TensorList tensors1,
    const at::TensorList tensors2,
    const at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars);
    // 暂不支持对scalarlist的处理，暂时走cuda原生路径
    return at::native::foreach_tensor_addcdiv_scalarlist_slow_(input, tensors1, tensors2, scalars);
}
}

