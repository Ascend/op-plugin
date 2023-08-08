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

#include <torch/csrc/autograd/custom_function.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace{
at::Tensor& cast_nocheck(at::Tensor& result, const at::Tensor& self) {
  int64_t dst_data_type = calcu_op_util::ConvertToAclDataType(result.scalar_type());
  at_npu::native::OpCommand cmd;
  cmd.Name("Cast")
      .Input(self)
      .Output(result)
      .Attr("dst_type", dst_data_type)
      .Run();
  return result;
}

at::Tensor npu_dtype_cast_impl(const at::Tensor& self, at::ScalarType dtype) {
  if (self.dtype() == dtype) {
    return self.clone();
  }
  at::Tensor result = npu_preparation::ApplyTensor(self.sizes(), self.options().dtype(dtype), self);
  cast_nocheck(result, self);
  return result;
}
} // namespace

at::Tensor& npu_dtype_cast_(at::Tensor& self, const at::Tensor& src) {
  if (self.dtype() == src.dtype()) {
    return self;
  }

  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_preparation::ApplyTensor(self);
    cast_nocheck(contiguous_self, src);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    cast_nocheck(self, src);
  }
  return self;
}

class NPUDtypeCastFunction : public torch::autograd::Function<NPUDtypeCastFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      at::Tensor self, 
      at::ScalarType dtype) {
    at::AutoNonVariableTypeMode g;
    ctx->saved_data["dtype"] = self.scalar_type();
    return npu_dtype_cast_impl(self, dtype);
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto dtype = ctx->saved_data["dtype"].toScalarType();
    grad_outputs[0].requires_grad_();
    return {NPUDtypeCastFunction::apply(grad_outputs[0], dtype), at::Tensor()};
  }
};

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype) {
  return NPUDtypeCastFunction::apply(self, dtype);
}
}  // namespace op_plugin
