// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_infer;

namespace {
constexpr int64_t kCrossVectorSize = 3;

inline void linalg_cross_check(const at::Tensor& self, const at::Tensor& other, int64_t dim)
{
    auto x_d = self.dim();
    auto y_d = other.dim();
    TORCH_CHECK(x_d == y_d, "linalg.cross: inputs must have the same number of dimensions.",
                OPS_ERROR(ErrCode::PARAM));
    auto wrap_dim = at::maybe_wrap_dim(dim, x_d);
    TORCH_CHECK(self.size(wrap_dim) == kCrossVectorSize && other.size(wrap_dim) == kCrossVectorSize,
                "linalg.cross: inputs dimension ", wrap_dim, " must have length ", kCrossVectorSize, ". Got ", self.size(wrap_dim),
                " and ", other.size(wrap_dim), OPS_ERROR(ErrCode::PARAM));
}
} // namespace

at::Tensor linalg_cross(const at::Tensor& self, const at::Tensor& other, int64_t dim)
{
    DO_COMPATIBILITY(aclnnLinalgCross, acl_op::linalg_cross(self, other, dim));
    linalg_cross_check(self, other, dim);
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, wrap_dim, out);
    return out;
}

at::Tensor& linalg_cross_out(const at::Tensor& self, const at::Tensor& other, int64_t dim, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnLinalgCross, acl_op::linalg_cross_out(self, other, dim, out));
    linalg_cross_check(self, other, dim);
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, wrap_dim, out);
    return out;
}
} // namespace op_api
