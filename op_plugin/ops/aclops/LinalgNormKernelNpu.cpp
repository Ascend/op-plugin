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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
#if VERSION_BETWEEN(V1R11, V1R11)
float calculate_p(at::Scalar p)
{
    float val = op_plugin::utils::get_scalar_float_value(p);
    if (val == INFINITY) {
        return static_cast<float>(INT_MAX); // p = inf
    } else if (val == -INFINITY) {
        return static_cast<float>(INT_MIN); // p = -inf
    } else {
        return static_cast<float>(val);
    }
}

at::Tensor& linalg_norm_out_npu_nocheck(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Scalar& ord,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype)
{
    at::Tensor fp32_self(self);
    if (self.scalar_type() != at::ScalarType::Float) {
        fp32_self = npu_dtype_cast(fp32_self, at::ScalarType::Float);
    }
    auto output_size = op_infer::reduce_ops_npu_output_size(fp32_self, dim, keepdim);
    at::Tensor result_temp = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
    at::Tensor result = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
    auto pvalue = calculate_p(ord);
    at_npu::native::OpCommand cmd1;
    cmd1.Name("LpNormReduceV2")
        .Input(fp32_self)
        .Output(result_temp)
        .Attr("p", pvalue)
        .Attr("axes", dim)
        .Attr("keepdim", keepdim)
        .Attr("epsilon", static_cast<float>(0))
        .Run();

    at_npu::native::OpCommand cmd2;
    cmd2.Name("LpNormUpdateV2")
        .Input(result_temp)
        .Output(result)
        .Attr("p", pvalue)
        .Attr("epsilon", static_cast<float>(0))
        .Run();
    // trans dtype for output
    if (result.scalar_type() != dtype) {
        result = npu_dtype_cast(result, dtype);
    }
    // until now, can not support resize shape of out correctly,
    // so the shape of out must be equal to output_size
    out = out.copy_(result);
    return out;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
static inline std::vector<int64_t> create_dim_backshift_permutation(int64_t dim0, int64_t dim1, int64_t ndim)
{
    TORCH_CHECK((dim0 != dim1) && (dim0 < ndim) && (dim0 >= 0) && (dim1 < ndim) && (dim1 >= 0),
                "duplicate or invalid dimensions" + OPS_ERROR(ErrCode::PARAM));
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

static void _linalg_matrix_norm_checks(const at::Tensor &A, std::vector<int64_t> &dim,
                                       at::optional<at::ScalarType> opt_dtype, bool low_precision = true)
{
    // A
    TORCH_CHECK(A.dim() >= 2, "linalg.matrix_norm", ": The input tensor ", "A", " must have at least 2 dimensions.", OPS_ERROR(ErrCode::PARAM));
    auto dtype = A.scalar_type();
    TORCH_CHECK((at::isFloatingType(dtype) || at::isComplexType(dtype)), "linalg.matrix_norm",
                ": Expected a floating point or complex tensor as input. Got ", dtype, OPS_ERROR(ErrCode::TYPE));
    if (!low_precision) {
        TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble || dtype == at::kComplexFloat ||
                        dtype == at::kComplexDouble,
                    "linalg.matrix_norm", ": Low precision dtypes not supported. Got ", dtype, OPS_ERROR(ErrCode::TYPE));
    }

    // dim
    TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple. Got ", dim, OPS_ERROR(ErrCode::PARAM));
    // wrap first to identify weird scenarios like A.ndim = 2, dim = (1, -1)
    // dim is modified in place while wrapping it
    at::maybe_wrap_dims(dim, A.dim());
    TORCH_CHECK(dim[0] != dim[1], "linalg.matrix_norm: dims must be different. Got (", dim[0], ", ", dim[1], ")", OPS_ERROR(ErrCode::PARAM));

    // dtype
    if (opt_dtype.has_value()) {
        auto self_dtype = A.scalar_type();
        auto dtype = opt_dtype.value();
        TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype), "linalg.matrix_norm",
                    ": dtype should"
                    " be floating point or complex, but got ",
                    dtype, OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(isComplexType(self_dtype) == isComplexType(dtype), "linalg.matrix_norm", ": dtype should be ",
                    isComplexType(self_dtype) ? "complex" : "real", " for ",
                    isComplexType(self_dtype) ? "complex" : "real", " inputs, but got ", dtype, OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(promoteTypes(self_dtype, dtype) == dtype, "linalg.matrix_norm", ": the dtype of the input ", "(",
                    self_dtype, ") should be convertible ", "without narrowing to the specified dtype (", dtype, ")", OPS_ERROR(ErrCode::TYPE));
    }
}

static inline std::vector<int64_t> create_reverse_permutation(std::vector<int64_t> permutation)
{
    int64_t ndim = static_cast<int64_t>(permutation.size());
    std::vector<int64_t> reverse_permutation(ndim);
    for (const auto dim_ind : c10::irange(ndim)) {
        reverse_permutation[permutation[dim_ind]] = dim_ind;
    }
    return reverse_permutation;
}

float calculate_p(at::Scalar p)
{
    float val = op_plugin::utils::get_scalar_float_value(p);
    if (val == INFINITY) {
        return static_cast<float>(INT_MAX); // p = inf
    } else if (val == -INFINITY) {
        return static_cast<float>(INT_MIN); // p = -inf
    } else {
        return static_cast<float>(val);
    }
}

at::Tensor &linalg_norm_out_npu_nocheck(at::Tensor &out, const at::Tensor &self, const at::Scalar &ord,
                                        at::IntArrayRef dim, bool keepdim, at::optional<at::ScalarType> dtype)
{
    at::Tensor fp32_self(self);
    if (self.scalar_type() != at::ScalarType::Float) {
        fp32_self = npu_dtype_cast(fp32_self, at::ScalarType::Float);
    }
    auto output_size = op_infer::reduce_ops_npu_output_size(fp32_self, dim, keepdim);
    at::Tensor result_temp = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
    at::Tensor result = npu_preparation::ApplyTensorWithSizes(output_size, fp32_self.options());
    auto pvalue = calculate_p(ord);
    at_npu::native::OpCommand cmd1;
    cmd1.Name("LpNormReduceV2")
        .Input(fp32_self)
        .Output(result_temp)
        .Attr("p", pvalue)
        .Attr("axes", dim)
        .Attr("keepdim", keepdim)
        .Attr("epsilon", static_cast<float>(0))
        .Run();

    at_npu::native::OpCommand cmd2;
    cmd2.Name("LpNormUpdateV2")
        .Input(result_temp)
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
    // so the shape of out must be equal to output_size
    out = out.copy_(result);
    return out;
}
#endif
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor linalg_vector_norm(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    c10::optional<at::IntArrayRef> opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype)
{
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
    at::Tensor out = npu_preparation::ApplyTensorWithSizes(output_size, self.options().dtype(dtype));
    linalg_norm_out_npu_nocheck(out, self, scalar_ord, dim, keepdim, dtype);
    return out;
}

at::Tensor& linalg_vector_norm_out(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    c10::optional<at::IntArrayRef> opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result)
{
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    npu_preparation::CheckOut(
        {self},
        result,
        ACL_FORMAT_ND,
        dtype,
        output_size);

    linalg_norm_out_npu_nocheck(result, self, scalar_ord, dim, keepdim, dtype);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor linalg_vector_norm(const at::Tensor &self, const at::Scalar &scalar_ord, at::OptionalIntArrayRef opt_dim,
                              bool keepdim, at::optional<at::ScalarType> opt_dtype)
{
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto self_val = opt_dtype.has_value() ? self.to(opt_dtype.value()) : self;
    at::Tensor out = npu_preparation::ApplyTensorWithSizes(output_size, self_val.options());
    linalg_norm_out_npu_nocheck(out, self_val, scalar_ord, dim, keepdim, opt_dtype);
    return out;
}

at::Tensor &linalg_vector_norm_out(const at::Tensor &self, const at::Scalar &scalar_ord,
                                   at::OptionalIntArrayRef opt_dim, bool keepdim,
                                   at::optional<at::ScalarType> opt_dtype, at::Tensor &result)
{
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    npu_preparation::CheckOut({self}, result, ACL_FORMAT_ND, self.scalar_type(), output_size);

    linalg_norm_out_npu_nocheck(result, self, scalar_ord, dim, keepdim, opt_dtype);
    return result;
}
#endif
} // namespace acl_op
