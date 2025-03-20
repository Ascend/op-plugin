// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace{
float calculate_p(c10::optional<at::Scalar> p)
{
    if (p.has_value()) {
        float val = op_plugin::utils::get_scalar_float_value(p.value());
        if (val == INFINITY) {
            return static_cast<float>(INT_MAX); // p = inf
        } else if (val == -INFINITY) {
            return static_cast<float>(INT_MIN); // p = -inf
        } else {
            return p.value().toFloat();
        }
    } else {
        return static_cast<float>(2.0); // default: p = 2.0
    }
}

inline bool check_use_aclop(float pfloat)
{
    if (pfloat != 0.0 && pfloat != 1.0 && pfloat != 2.0 && pfloat != 3.0) {
        if (op_plugin::utils::is_gte_cann_version_810rc1() &&
            (pfloat == static_cast<float>(INT_MAX) || pfloat == static_cast<float>(INT_MIN))) {
            // Version 8.1.RC1 of cann began to support norm operators with p being inf or -inf
            return false;
        }
        return true;
    }
    return false;
}

inline at::Tensor &norm_out_npu_nocheck_opapi(at::Tensor &out,
                                              const at::Tensor &self,
                                              c10::optional<at::Scalar> p,
                                              at::IntArrayRef dim,
                                              bool keepdim)
{
    at::Scalar pvalue = 2;
    if (p.has_value()) {
        pvalue = p.value();
    }
    EXEC_NPU_CMD(aclnnNorm, self, pvalue, dim, keepdim, out);
    return out;
}

inline at::Tensor &norm_out_imp(const at::Tensor &self,
                                const c10::optional<at::Scalar> &p,
                                at::IntArrayRef dim,
                                bool keepdim,
                                at::ScalarType dtype,
                                at::Tensor &out)
{
    float pfloat = calculate_p(p);
    if (check_use_aclop(pfloat)) {
        return acl_op::norm_out(self, p, dim, keepdim, dtype, out);
    } else {
        auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
        npu_preparation::check_tensor({self}, out, dtype, outputSize);
        return norm_out_npu_nocheck_opapi(out, self, p, dim, keepdim);
    }
}

inline at::Tensor norm_imp(const at::Tensor &self,
                           const c10::optional<at::Scalar> &p,
                           at::IntArrayRef dim,
                           bool keepdim,
                           at::ScalarType dtype)
{
    float pfloat = calculate_p(p);
    if (check_use_aclop(pfloat)) {
        return acl_op::norm(self, p, dim, keepdim, dtype);
    } else {
        auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
        auto dtype_checked = dtype;
        if (self.is_complex()) {
            dtype_checked = self.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
        }
        at::Tensor out = npu_preparation::apply_tensor_with_sizes(outputSize, self.options().dtype(dtype_checked));
        return norm_out_npu_nocheck_opapi(out, self, p, dim, keepdim);
    }
}
} // namespace

// norm.dtype_out
at::Tensor& norm_out(const at::Tensor &self,
                     const c10::optional<at::Scalar> &p,
                     at::IntArrayRef dim,
                     bool keepdim,
                     at::ScalarType dtype,
                     at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm_out(self, p, dim, keepdim, dtype, out));
    return norm_out_imp(self, p, dim, keepdim, out.scalar_type(), out);
}

// norm.out
at::Tensor& norm_out(const at::Tensor &self,
                     const c10::optional<at::Scalar> &p,
                     at::IntArrayRef dim,
                     bool keepdim,
                     at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm_out(self, p, dim, keepdim, out));
    return norm_out_imp(self, p, dim, keepdim, out.scalar_type(), out);
}

// norm.ScalarOpt_dim_dtype
at::Tensor norm(const at::Tensor &self,
                const c10::optional<at::Scalar> &p,
                at::IntArrayRef dim,
                bool keepdim,
                at::ScalarType dtype)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm(self, p, dim, keepdim, dtype));
    return norm_imp(self, p, dim, keepdim, dtype);
}

// norm.ScalarOpt_dtype
at::Tensor norm(const at::Tensor &self,
                const c10::optional<at::Scalar> &p,
                at::ScalarType dtype)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm(self, p, dtype));
    return norm_imp(self, p, {}, false, dtype);
}

// norm.Scalar
at::Tensor norm(const at::Tensor &self,
                const at::Scalar &p)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm(self, p));
    return norm_imp(self, p, {}, false, self.scalar_type());
}

// norm.ScalarOpt_dim
at::Tensor norm(const at::Tensor &self,
                const c10::optional<at::Scalar> &p,
                at::IntArrayRef dim,
                bool keepdim)
{
    DO_COMPATIBILITY(aclnnNorm, acl_op::norm(self, p, dim, keepdim));
    return norm_imp(self, p, dim, keepdim, self.scalar_type());
}

}
