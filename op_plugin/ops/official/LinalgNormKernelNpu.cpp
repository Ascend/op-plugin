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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
static inline std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim) {
  TORCH_CHECK(
    (dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
    "duplicate or invalid dimensions");
  std::vector<int64_t> permutation(ndim);
  int64_t cur_permuted_dim = 0;
  for (const auto dim_ind : c10::irange(ndim)) {
    if ((dim_ind != dim0) && (dim_ind != dim1)) {
      permutation[cur_permuted_dim++] = dim_ind;
    }
  }
  permutation[cur_permuted_dim++] = dim0;
  permutation[cur_permuted_dim] = dim1;
  return permutation;
}

static void _linalg_matrix_norm_checks(
    const at::Tensor& A,
    std::vector<int64_t>& dim,
    at::optional<at::ScalarType> opt_dtype,
    bool low_precision = true) {
  // A
  TORCH_CHECK(A.dim() >= 2, "linalg.matrix_norm", ": The input tensor ", "A", " must have at least 2 dimensions.");
  auto dtype = A.scalar_type();
  TORCH_CHECK((at::isFloatingType(dtype) || at::isComplexType(dtype)),
              "linalg.matrix_norm", ": Expected a floating point or complex tensor as input. Got ", dtype);
  if (!low_precision) {
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble || dtype == at::kComplexFloat || dtype == at::kComplexDouble,
                "linalg.matrix_norm", ": Low precision dtypes not supported. Got ", dtype);
  }

  // dim
  TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple. Got ", dim);
  // wrap first to identify weird scenarios like A.ndim = 2, dim = (1, -1)
  // dim is modified in place while wrapping it
  at::maybe_wrap_dims(dim, A.dim());
  TORCH_CHECK(dim[0] != dim[1], "linalg.matrix_norm: dims must be different. Got (", dim[0], ", ", dim[1], ")");

  // dtype
  if (opt_dtype.has_value()) {
    auto self_dtype = A.scalar_type();
    auto dtype = opt_dtype.value();
    TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype), "linalg.matrix_norm", ": dtype should"
        " be floating point or complex, but got ", dtype);
    TORCH_CHECK(isComplexType(self_dtype) == isComplexType(dtype),
        "linalg.matrix_norm", ": dtype should be ", isComplexType(self_dtype) ? "complex" : "real",
        " for ", isComplexType(self_dtype) ? "complex" : "real", " inputs, but got ", dtype);
    TORCH_CHECK(promoteTypes(self_dtype, dtype) == dtype,
        "linalg.matrix_norm", ": the dtype of the input ", "(", self_dtype, ") should be convertible ",
        "without narrowing to the specified dtype (", dtype, ")");
  }
}

static inline std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation) {
  int64_t ndim = permutation.size();
  std::vector<int64_t> reverse_permutation(ndim);
  for (const auto dim_ind : c10::irange(ndim)) {
    reverse_permutation[permutation[dim_ind]] = dim_ind;
  }
  return reverse_permutation;
}

float calculate_p(at::Scalar p) {
    float val = op_plugin::utils::get_scalar_float_value(p);
    if (val == INFINITY) {
        return static_cast<float>(INT_MAX); // p = inf
    } else if (val == -INFINITY) {
        return static_cast<float>(INT_MIN); // p = -inf
    } else {
        return static_cast<float>(val);
    }
}

at::Tensor &linalg_norm_out_npu_nocheck(
    at::Tensor &out,
    const at::Tensor &self,
    const at::Scalar& ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::optional<at::ScalarType> dtype) {
  at::Tensor fp32Self(self);
  if (self.scalar_type() != at::ScalarType::Float) {
    fp32Self = npu_dtype_cast(fp32Self, at::ScalarType::Float);
  }
  auto outputSize = op_infer::reduce_ops_npu_output_size(fp32Self, dim, keepdim);
  if (outputSize.empty()){
    outputSize.push_back(1);
  }
  at::Tensor resultTemp = npu_preparation::ApplyTensorWithSizes(outputSize, fp32Self.options());
  at::Tensor result = npu_preparation::ApplyTensorWithSizes(outputSize, fp32Self.options());
  auto pvalue = calculate_p(ord);
  at_npu::native::OpCommand cmd1;
  cmd1.Name("LpNormReduceV2")
      .Input(fp32Self)
      .Output(resultTemp)
      .Attr("p", pvalue)
      .Attr("axes", dim)
      .Attr("keepdim", keepdim)
      .Attr("epsilon", static_cast<float>(0))
      .Run();

  at_npu::native::OpCommand cmd2;
  cmd2.Name("LpNormUpdateV2")
      .Input(resultTemp)
      .Output(result)
      .Attr("p", pvalue)
      .Attr("epsilon", static_cast<float>(0))
      .Run();
  // trans dtype for output
  if (result.scalar_type() != dtype) {
    auto dtype_ = dtype.value_or(self.scalar_type());
    result = npu_dtype_cast(result, dtype_);
  }
  // until now, can not support resize shape of out correctly,
  // so the shape of out must be equal to outputSize
  out = out.copy_(result);
  return out;
}
} // namespace

at::Tensor linalg_vector_norm(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype) {
  auto dim = opt_dim.value_or(at::IntArrayRef{});
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  auto self_ = opt_dtype.has_value() ? self.to(opt_dtype.value()) : self;
  at::Tensor out = npu_preparation::ApplyTensorWithSizes(output_size, self_.options());
  linalg_norm_out_npu_nocheck(out, self_, scalar_ord, dim, keepdim, opt_dtype);
  return out;
}

at::Tensor& linalg_vector_norm_out(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result) {
  auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<at::IntArrayRef::value_type>{0, 1};
  auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
  npu_preparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      output_size);

  result = op_plugin::linalg_vector_norm(self, scalar_ord, opt_dim, keepdim, opt_dtype);
  return result;
}

at::Tensor& linalg_matrix_norm_out(
    const at::Tensor& A,
    const at::Scalar& ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(A, dim, keepdim);
  npu_preparation::CheckOut(
      {A},
      result,
      ACL_FORMAT_ND,
      A.scalar_type(),
      output_size);

  result = op_plugin::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  return result;
}

at::Tensor& linalg_matrix_norm_out(
    const at::Tensor& A,
    c10::string_view ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(A, dim, keepdim);
  npu_preparation::CheckOut(
      {A},
      result,
      ACL_FORMAT_ND,
      A.scalar_type(),
      output_size);

  result = op_plugin::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  return result;
}

at::Tensor linalg_matrix_norm(
    const at::Tensor& A,
    const at::Scalar& scalar_ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype) {
  // Check ord first as it will be used in the dtype check of A
  auto ord = scalar_ord.toDouble();
  auto abs_ord = std::abs(ord);
  TORCH_CHECK(abs_ord == 2. || abs_ord == 1. || abs_ord == INFINITY, "linalg.matrix_norm: Order ", ord, " not supported.");

  auto dim_ = dim.vec();
  // Check A, dim, and dtype
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, abs_ord != 2.);

  auto max_min = [ord, keepdim](const at::Tensor& A, int64_t dim) { return ord > 0 ? A.amax(dim, keepdim) : A.amin(dim, keepdim); };
  if (abs_ord == 2.) {
    // Move dims to the end
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A.dim());

    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;
    auto result = max_min(op_plugin::linalg_svdvals(A_.permute(permutation), ""), -1);
    if (keepdim) {
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;
  } else {  // 1, -1, inf, -inf
    // The infty norm is like the 1 norm on the transposed matrix
    if (abs_ord == INFINITY) {
      std::swap(dim_[0], dim_[1]);
    }

    // If the first reduction removes one dim from the front (dim_[0] < dim_[1]), after this
    // reduction dim_[1] will be off by one
    if (!keepdim && (dim_[0] < dim_[1])) {
      dim_[1]--;
    }
    return max_min(op_plugin::linalg_vector_norm(A, 1., {dim_[0]}, keepdim, opt_dtype), dim_[1]);
  }
}

// fro / nuc
at::Tensor linalg_matrix_norm(
    const at::Tensor& A,
    c10::string_view ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype) {
  // Check ord first as it will be used in the dtype check of A
  TORCH_CHECK(ord == "fro" || ord == "nuc", "linalg.matrix_norm: Order ", ord, " not supported.");

  auto dim_ = dim.vec();
  // Check A, dim, and dtype
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, ord != "nuc");

  if (ord == "fro") {
    return op_plugin::linalg_vector_norm(A, 2, dim_, keepdim, opt_dtype);
  } else {  // nuc
    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;

    // Move dims to the end
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A_.dim());
    auto result = op_plugin::linalg_svdvals(A_.permute(permutation), "").sum(-1, keepdim);
    if (keepdim) {
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;
  }
}

at::Tensor& linalg_norm_out(
    const at::Tensor& X,
    const at::optional<at::Scalar>& opt_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(X, opt_dim.value_or(at::IntArrayRef{}), keepdim);
  npu_preparation::CheckOut(
      {X},
      result,
      ACL_FORMAT_ND,
      X.scalar_type(),
      output_size);

  result = op_plugin::linalg_norm(X, opt_ord, opt_dim, keepdim, opt_dtype);
  return result;
}

at::Tensor& linalg_norm_out(
    const at::Tensor& X,
    c10::string_view ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result) {
  auto output_size = op_infer::reduce_ops_npu_output_size(X, opt_dim.value_or(at::IntArrayRef{}), keepdim);
  npu_preparation::CheckOut(
      {X},
      result,
      ACL_FORMAT_ND,
      X.scalar_type(),
      output_size);

  result = op_plugin::linalg_norm(X, ord, opt_dim, keepdim, opt_dtype);
  return result;
}

at::Tensor linalg_norm(
    const at::Tensor& X,
    const at::optional<at::Scalar>& opt_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype) {
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim ->size() == 2, "linalg.norm: If ",
              "dim is specified, it must be of length 1 or 2. Got ", *opt_dim);
  } else {
    if (opt_ord.has_value()) {
      TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                  "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
    }
  }

  // If ord=None, we'll always use the 2-norm or frob norm (which are the same) so we go through
  // vector_norm
  if (opt_ord.has_value() &&
       ((opt_dim.has_value() && opt_dim->size() == 2) ||
        (!opt_dim.has_value() && X.dim() == 2))) {
    auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<at::IntArrayRef::value_type>{0, 1};
    return op_plugin::linalg_matrix_norm(X, *opt_ord, dim, keepdim, opt_dtype);
  } else {
    auto scalar_ord = opt_ord.value_or(at::Scalar(2.));
    return op_plugin::linalg_vector_norm(X, scalar_ord, opt_dim, keepdim, opt_dtype);
  }
}

at::Tensor linalg_norm(
    const at::Tensor& X,
    c10::string_view ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype) {
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim ->size() == 2, "linalg.norm: If ",
              "dim is specified, it mut be of length 1 or 2. Got ", *opt_dim);
  } else {
    TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
  }
  auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<at::IntArrayRef::value_type>{0, 1};
  return op_plugin::linalg_matrix_norm(X, ord, dim, keepdim, opt_dtype);
}
} // namespace op_plugin