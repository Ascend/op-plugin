// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NOTE (rms_norm design v1.1): this shell calls the CANN official
// aclnnRmsNormGrad interface DIRECTLY (no reuse of the custom op
// op_api::npu_rms_norm_backward) for aten-path independence, and aligns
// capability/edge behavior with _fused_rms_norm_backward_cuda.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <limits>

#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)

namespace op_api {

namespace {

inline at::ScalarType rms_norm_opmath_dtype(at::ScalarType t)
{
    switch (t) {
        case at::kHalf:
        case at::kBFloat16:
        case at::kFloat:
            return at::kFloat;
        default:
            return t;
    }
}

// Composite-equivalent rms_norm backward, written with public aten ops.
// torch::autograd::infinitely_differentiable_native_rms_norm_backward is NOT
// exported from libtorch_cpu (torch/csrc/autograd/FunctionsManual.h declares it
// without TORCH_API; dlopen probe shows undefined symbol), so its dY path is
// mirrored here (FunctionsManual.cpp:5534-5594, with drstd = undefined since
// the first-order backward shell receives no rstd gradient):
//   grad_x_hat = dY * weight                     (:5535-5536)
//   dX = rstd*grad_x_hat - (X*rstd^3/N) * sum(X*grad_x_hat, ns_dims, keepdim)
//                                                (:5553-5557)
//   dgamma = sum(dY * X * rstd, outer_dims)      (:5577-5588)
std::tuple<at::Tensor, at::Tensor> rms_norm_composite_backward_fallback(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& rstd,
    const at::Tensor& weight,
    std::array<bool, 2> output_mask)
{
    const int64_t input_ndim = input.dim();
    const int64_t normalized_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t axis = input_ndim - normalized_ndim;
    int64_t n_rms = 1;
    for (const auto i : c10::irange(normalized_ndim)) {
        n_rms *= normalized_shape[i];
    }

    const at::ScalarType opmath = rms_norm_opmath_dtype(input.scalar_type());
    at::Tensor dY = grad_out.to(opmath);
    at::Tensor x = input.to(opmath);
    at::Tensor rstd_opmath = rstd.to(opmath);
    at::Tensor weight_opmath = weight.to(opmath);

    // rstd arrives with the forward's stat_shape (trailing normalized dims set
    // to 1), which broadcasts against input directly (equivalent to the
    // rstd.view(rstd_view_shape) of FunctionsManual.cpp:5524-5530)
    at::Tensor rstd_pow3 = rstd_opmath.pow(3);
    at::Tensor grad_x_hat = dY * weight_opmath;  // weight is always defined here

    std::vector<int64_t> inner_sum_dims;
    inner_sum_dims.reserve(normalized_ndim);
    for (const auto i : c10::irange(normalized_ndim)) {
        inner_sum_dims.push_back(axis + i);
    }

    at::Tensor dx_out;
    if (output_mask[0]) {
        at::Tensor sum_input_times_grad_x_hat =
            at::sum(x * grad_x_hat, at::IntArrayRef(inner_sum_dims), /*keepdim=*/true);
        at::Tensor dx = rstd_opmath * grad_x_hat -
            (x * rstd_pow3 / static_cast<double>(n_rms)) * sum_input_times_grad_x_hat;
        dx_out = dx.to(grad_out.scalar_type());
    }

    at::Tensor dgamma_out = at::Tensor();
    if (output_mask[1]) {
        at::Tensor dgamma_full_shape = dY * (x * rstd_opmath);  // dY * x_hat
        at::Tensor dgamma;
        if (axis > 0) {
            std::vector<int64_t> outer_sum_dims;
            outer_sum_dims.reserve(axis);
            for (const auto i : c10::irange(axis)) {
                outer_sum_dims.push_back(i);
            }
            dgamma = at::sum(dgamma_full_shape, at::IntArrayRef(outer_sum_dims), /*keepdim=*/false);
        } else {
            dgamma = dgamma_full_shape;
        }
        dgamma_out = dgamma.to(weight.scalar_type());
    }
    return std::tuple<at::Tensor, at::Tensor>(dx_out, dgamma_out);
}

} // namespace

std::tuple<at::Tensor, at::Tensor> _fused_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    std::array<bool, 2> output_mask) {
  TORCH_CHECK(!normalized_shape.empty() && normalized_shape.size() <= (size_t)input.dim(),
              "rms_norm(): normalized_shape must have 1 ~ input.dim() elements, but got ",
              normalized_shape.size());
  if (weight_opt.has_value() && weight_opt->defined()) {
    TORCH_CHECK(weight_opt->sizes().equals(normalized_shape),
                "Expected weight to be of same shape as normalized_shape, but got ",
                "weight of shape ", weight_opt->sizes(),
                " and normalized_shape = ", normalized_shape);
  }
  TORCH_CHECK(
      input.sizes().slice(input.dim() - normalized_shape.size()).equals(normalized_shape),
      "Given normalized_shape=", normalized_shape,
      ", expected input with shape [*, ", normalized_shape,
      "], but got input of size ", input.sizes());
  at::Tensor weight = (weight_opt.has_value() && weight_opt->defined())
      ? *weight_opt : at::ones(normalized_shape, input.options());
  if (!is_ascend950_path() || input.scalar_type() == at::kDouble) {
    return rms_norm_composite_backward_fallback(
        grad_out, input, normalized_shape, rstd, weight, output_mask);
  }
  bool need_fold = (weight.dim() > 1) || (input.dim() == 1);
  at::Tensor dy = need_fold ? grad_out.reshape({-1, weight.numel()}) : grad_out;
  at::Tensor x = need_fold ? input.reshape({-1, weight.numel()}) : input;
  at::Tensor gamma = weight.reshape({-1});
  const int64_t M = x.size(0);
  dy = *dy.expect_contiguous();
  x = *x.expect_contiguous();
  gamma = *gamma.expect_contiguous();
  at::Tensor rstd_kernel = need_fold ? rstd.reshape({M, 1}) : rstd;
  at::Tensor dx = at::empty_like(x);
  // aclnnRmsNormGrad accumulates dgamma in fp32 (official dtype matrix:
  // dgamma is always FLOAT32).
  at::Tensor dgamma = at::empty({gamma.numel()}, x.options().dtype(at::kFloat));

  if (x.numel() > 0) {
    EXEC_NPU_CMD(aclnnRmsNormGrad, dy, x, rstd_kernel, gamma, dx, dgamma);
  }
  at::Tensor dx_out;
  if (output_mask[0]) {
    dx_out = need_fold ? dx.reshape(input.sizes()) : dx;
  }
  at::Tensor dgamma_out = at::Tensor();
  if (output_mask[1]) {
    at::Tensor dg = (x.numel() > 0) ? dgamma.to(weight.scalar_type())
                                    : at::zeros_like(weight);
    if (need_fold) {
      dg = dg.reshape(weight.sizes());
    }
    dgamma_out = dg;
  }
  return std::tuple<at::Tensor, at::Tensor>(dx_out, dgamma_out);
}

} // namespace op_api
#endif
