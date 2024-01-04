#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_layer_norm(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    double epsilon,
    bool additional_output)
{
    DO_COMPATIBILITY(aclnnAddLayerNorm, acl_op::npu_add_layer_norm(x1, x2, gamma, beta, epsilon, additional_output));
    at::SmallVector<int64_t, SIZE> shape;
    for (uint64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        shape.emplace_back(x1.size(index));
    }
    shape.emplace_back(1);
    
    at::Tensor y = npu_preparation::apply_tensor(x1);
    at::Tensor x = npu_preparation::apply_tensor(x1);
    at::Tensor mean = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor rstd = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    const at::Tensor& bias = at::Tensor();
    
    EXEC_NPU_CMD(aclnnAddLayerNorm, x1, x2, gamma, beta, bias, epsilon, additional_output, y, mean, rstd, x);
    return std::make_tuple(y, mean, rstd, x);
}
} // namespace op_api