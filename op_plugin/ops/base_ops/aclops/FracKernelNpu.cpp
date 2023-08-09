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

namespace{
at::Tensor& frac_out_nocheck(at::Tensor& result, const at::Tensor& self) {
  at::Tensor cast_return_Tensor = op_plugin::npu_dtype_cast(self, at::kInt);
  at::sub_out(result, self, cast_return_Tensor);
  return result;
}
} // namespace

at::Tensor& frac_out(const at::Tensor& self, at::Tensor& result) {
  npu_preparation::CheckOut(
      {self}, 
      result,
      self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    frac_out_nocheck(contiguous_result, self);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    frac_out_nocheck(result, self);
  }
  return result;
}

at::Tensor frac(const at::Tensor& self) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  frac_out_nocheck(result, self);
  return result;
}

at::Tensor& frac_(at::Tensor& self) {
  return op_plugin::frac_out(self, self);
}
} // namespace op_plugin
