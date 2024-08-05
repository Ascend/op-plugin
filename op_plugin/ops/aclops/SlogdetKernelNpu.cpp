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
std::tuple<at::Tensor&, at::Tensor&> slogdet_out_nocheck(
    at::Tensor& sign,
    at::Tensor& result,
    const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("LogMatrixDeterminant")
      .Input(self)
      .Output(sign)
      .Output(result)
      .Run();

  return std::tie(sign, result);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> slogdet(const at::Tensor& self) {
    TORCH_CHECK(self.dim() >= 2, "input must be at least 2 dimensions" + OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.erase(output_size.end() - 2, output_size.end());
    at::Tensor sign = npu_preparation::apply_tensor(self, output_size);
    at::Tensor y = npu_preparation::apply_tensor(self, output_size);

    slogdet_out_nocheck(sign, y, self);

    return std::tie(sign, y);
}

#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor, at::Tensor> linalg_slogdet(const at::Tensor& self)
{
    TORCH_CHECK(self.dim() >= 2, "input must be at least 2 dimensions", OPS_ERROR(ErrCode::PARAM));

    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.erase(output_size.end() - 2, output_size.end());

    at::Tensor sign = npu_preparation::apply_tensor(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);

    slogdet_out_nocheck(sign, result, self);
    return std::tie(sign, result);
}

std::tuple<at::Tensor&, at::Tensor&> linalg_slogdet_out(
    const at::Tensor& self,
    at::Tensor& sign,
    at::Tensor& result)
{
    TORCH_CHECK(self.dim() >= 2, "input must be at least 2 dimensions", OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.erase(output_size.end() - 2, output_size.end());
    npu_preparation::CheckOut(
        {self},
        sign,
        sign,
        output_size);
    npu_preparation::CheckOut(
        {self},
        result,
        result,
        output_size);

    slogdet_out_nocheck(sign, result, self);
    bool sign_match = npu_utils::check_match(&sign);
    bool result_match = npu_utils::check_match(&result);
    if (!(sign_match && result_match)) {
        at::Tensor contiguous_sign = sign_match ? sign : npu_utils::format_contiguous(sign);
        at::Tensor contiguous_result = result_match ? result : npu_utils::format_contiguous(result);
        slogdet_out_nocheck(contiguous_sign, contiguous_result, self);
        if (!sign_match) {
            npu_utils::format_fresh_view(sign, contiguous_sign);
        }
        if (!result_match) {
            npu_utils::format_fresh_view(result, contiguous_result);
        }
    } else {
        slogdet_out_nocheck(sign, result, self);
    }
    return std::tie(sign, result);
}
#endif
} // namespace acl_op
