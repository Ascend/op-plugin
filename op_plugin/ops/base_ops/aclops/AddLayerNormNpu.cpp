#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_layer_norm(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma, const at::Tensor &beta, double epsilon, bool additional_output)
{
    at::SmallVector<int64_t, SIZE> shape;
    for (uint64_t index = 0; index < x1.dim() - gamma.dim(); index++) {
        shape.emplace_back(x1.size(index));
    }
    shape.emplace_back(1);
    
    at::Tensor y = npu_preparation::apply_tensor(x1);
    at::Tensor x = npu_preparation::apply_tensor(x1);
    at::Tensor mean = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor rstd = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at_npu::native::OpCommand cmd;
    cmd.Name("AddLayerNorm")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(gamma, "gamma")
        .Input(beta, "beta")
        .Output(y, "y")
        .Output(mean, "mean")
        .Output(rstd, "rstd")
        .Output(x, "x")
        .Attr("epsilon", static_cast<float>(epsilon))
        .Attr("additional_output", additional_output)
        .Run();
    return std::make_tuple(y, mean, rstd, x);
}
} // namespace acl_op