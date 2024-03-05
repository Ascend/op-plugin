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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> renorm_npu_output_size(
    const at::Tensor& self,
    int64_t dim) {
  c10::SmallVector<int64_t, SIZE> out_size;
  for (int64_t i = 0; i < self.dim(); i++) {
    if (i != dim) {
      out_size.emplace_back(1);
    } else {
      out_size.emplace_back(self.sizes()[i]);
    }
  }
  return out_size;
}

at::Tensor& renorm_compute(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar p,
    int64_t dim,
    at::Scalar maxnorm) {
  float p_value = op_plugin::utils::get_scalar_float_value(p);
  float maxnorm_value = op_plugin::utils::get_scalar_float_value(maxnorm);

  at_npu::native::OpCommand cmd;
  cmd.Name("Renorm")
      .Input(self)
      .Output(result)
      .Attr("p", p_value)
      .Attr("maxnorm", maxnorm_value)
      .Attr("dim", dim)
      .Run();
  return result;
}

at::Tensor& renorm_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar p,
    int64_t dim,
    at::Scalar maxnorm) {
    auto ori_type = self.scalar_type();
    if (ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
        TORCH_CHECK(false, "Renorm only support float16 or float32 type." + OPS_ERROR(ErrCode::TYPE));
    }

    TORCH_CHECK(result.scalar_type() == ori_type, "result's type must be equal to input's."
        + OPS_ERROR(ErrCode::TYPE));

    dim = op_plugin::utils::make_warp_dim(dim, self.dim());
    auto output_size = renorm_npu_output_size(self, dim);
    at::Tensor result_bak = npu_preparation::apply_tensor_with_format(
        output_size,
        self.options().dtype(at::kFloat),
        npu_preparation::get_tensor_npu_format(self));
    if (ori_type == c10::ScalarType::Half) {
        at::Tensor self_no_name = self.rename(c10::nullopt);
        at::Tensor result_no_name = result.rename(c10::nullopt);
        self_no_name = at_npu::native::custom_ops::npu_dtype_cast(self_no_name, c10::ScalarType::Float);
        result_no_name = at_npu::native::custom_ops::npu_dtype_cast(result_no_name, c10::ScalarType::Float);
        renorm_compute(
            result_bak,
            self_no_name,
            p,
            dim,
            maxnorm);

        at::Tensor result_broadcast = acl_op::npu_broadcast(result_bak, self.sizes());
        at::mul_out(result_no_name, result_broadcast, self_no_name);
        acl_op::npu_dtype_cast_(result, result_no_name);
    } else {
        renorm_compute(
            result_bak,
            self,
            p,
            dim,
            maxnorm);

        at::Tensor result_broadcast = acl_op::npu_broadcast(result_bak, self.sizes());
        at::mul_out(result, result_broadcast, self);
    }
    return result;
}
} // namespace

at::Tensor& renorm_out(
    const at::Tensor& self,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    renorm_out_nocheck(contiguous_self, contiguous_self, p, dim, maxnorm);
    npu_utils::format_fresh_view(result, contiguous_self);
  } else {
    renorm_out_nocheck(result, self, p, dim, maxnorm);
  }

  return result;
}

at::Tensor renorm(const at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  renorm_out_nocheck(result, self, p, dim, maxnorm);
  return result;
}

at::Tensor& renorm_(at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm) {
    return acl_op::renorm_out(self, p, dim, maxnorm, self);
}
} // namespace acl_op
