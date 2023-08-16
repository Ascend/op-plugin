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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using torch::autograd::AutogradContext;
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
std::tuple<at::Tensor&, at::Tensor&> max_v1_out_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}

std::tuple<at::Tensor, at::Tensor> npu_max_cal(const at::Tensor& self, int64_t dim, bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  c10::SmallVector<int64_t, SIZE> output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indices_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = output_size.empty() ? ACL_FORMAT_NCHW : calcu_op_util::GetTensorNpuFormat(self);

  at::Tensor outputs = npu_preparation::apply_tensor_with_format(
      output_size, self.options(), npu_format);
  at::Tensor indices = npu_preparation::apply_tensor_with_format(
      indices_size, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);
  max_v1_out_nocheck(outputs, indices, self, dim, keepdim);
  return std::tie(outputs, indices);
}
} // namespace

at::Tensor npu_max_backward(
    const at::Tensor& grad,
    int64_t dim,
    const at::Tensor& indices,
    at::IntArrayRef sizes,
    bool keepdim) {
  at::Tensor new_grad = grad;
  at::Tensor new_indices = indices;
  if (keepdim && sizes.size() > 0) {
    new_grad = grad.squeeze(dim);
    new_indices = indices.squeeze(dim);
  }
  if (new_indices.dtype() == at::kLong) {
    new_indices = acl_op::npu_dtype_cast(new_indices, at::kInt);
  }
  auto grad_input = acl_op::npu_scatter(at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
  return grad_input;
}

class NPUMaxFunction : public torch::autograd::Function<NPUMaxFunction> {
public:
  static std::vector<at::Tensor> forward(AutogradContext *ctx,
      const at::Tensor& self,
      int64_t dim,
      bool keepdim) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["shape"] = self.sizes();
    ctx->saved_data["keepdim"] = keepdim;
    at::AutoNonVariableTypeMode g;
    auto result = npu_max_cal(self, dim, keepdim);
    auto indices = std::get<1>(result);
    ctx->save_for_backward({indices});
    std::vector<at::Tensor> result_list = {std::get<0>(result), indices};
    return result_list;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto dim = ctx->saved_data["dim"].toInt();
    auto sizes = ctx->saved_data["shape"].toIntVector();
    auto keepdim = ctx->saved_data["keepdim"].toBool();
    auto saved = ctx->get_saved_variables();
    auto indices = saved[0];
    at::Tensor result = acl_op::npu_max_backward(grad_outputs[0], dim, indices, sizes, keepdim);

    std::vector<at::Tensor> output = {result, at::Tensor(), at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, int64_t dim, bool keepdim) {
  auto output = NPUMaxFunction::apply(self, dim, keepdim);
  std::tuple<at::Tensor, at::Tensor> result(output[0], output[1]);
  return result;
}

std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return acl_op::npu_max(self, dimname_to_position(self, dim), keepdim);
}
} // namespace acl_op
