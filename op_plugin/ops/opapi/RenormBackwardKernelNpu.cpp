#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include <numeric>
#include <utility>
#include <optional>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor norm_backward_impl(
    at::Tensor grad,
    const at::Tensor& self,
    const std::optional<at::Scalar>& p_,
    at::Tensor norm,
    at::IntArrayRef dim,
    bool /* keepdim */) {

    double p = p_.value_or(2.0).toDouble();
    at::Tensor self_scaled;
    at::Tensor scale_v;

    if (p == 0.0) {
        return {};
    } else if (p == 1.0) {
        return self.sgn() * grad;
    } else if (p == 2.0) {
        return grad * (self / norm).masked_fill_(norm == 0, 0);
    } else if (std::isinf(p)) {
        auto self_abs = self.abs();
        auto mask = self_abs.eq(norm).logical_or(self_abs.isnan());
        return self.sgn() * ((grad / mask.sum(dim, true)) * mask);
    } else if (p < 1.0) {
        self_scaled =
            self.sgn() * self.abs().pow_(p - 1).masked_fill_(self == 0, 0);
        return self_scaled * grad * norm.pow(1 - p);
    } else if (p < 2.0) {
        self_scaled = self.sgn() * self.abs().pow_(p - 1);
        scale_v = grad / norm.pow(p - 1);
        scale_v.masked_fill_(norm == 0, 0);
        return self_scaled * scale_v;
    } else {
        self_scaled = self * self.abs().pow_(p - 2);
        scale_v = grad / norm.pow(p - 1);
        scale_v.masked_fill_(norm == 0, 0);
        return self_scaled * scale_v;
    }
}

at::Tensor npu_renorm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm) {

    auto n = self.dim();
    dim = c10::maybe_wrap_dim(dim, n);
    auto reduce_dims = at::DimVector(n);
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
    reduce_dims.erase(reduce_dims.begin() + dim);

    // NPU: BFloat16/Half/float → float
    auto scalar_type = self.scalar_type();
    auto acc_type = (scalar_type == c10::ScalarType::BFloat16 ||
                     scalar_type == c10::ScalarType::Half ||
                     scalar_type == c10::ScalarType::Float)
                    ? c10::ScalarType::Float
                    : scalar_type;

    auto norm = op_api::linalg_vector_norm(
        self, p, reduce_dims, /*keepdim=*/true, /*dtype=*/acc_type);

    const auto real_acc_type = c10::toRealValueType(acc_type);
    auto grad_output = (self.conj() * grad);

    if (real_acc_type != acc_type) {
        grad_output = at::real(grad_output);
    }

    grad_output = grad_output.sum(
        reduce_dims, /*keepdim=*/true, /*dtype=*/real_acc_type);

    std::optional<at::Scalar> opt_p = p;
    auto nb = norm_backward_impl(
        std::move(grad_output), self, opt_p, norm, reduce_dims, /*keepdim=*/true);

    auto invnorm = (norm + 1e-7).reciprocal();
    auto grad_norm = maxnorm * invnorm * (grad - invnorm * nb);

    return at::where(norm > maxnorm, grad_norm.to(grad.scalar_type()), grad);
}

} // namespace op_api
