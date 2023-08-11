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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nll_loss2d_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    int64_t ignore_index) {
  c10::SmallVector<int64_t, SIZE> output_size;
  c10::SmallVector<int64_t, SIZE> total_weight_size;

  if (reduction == at::Reduction::None) {
    output_size = {self.size(0)};
  }

  return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
      output_size, total_weight_size);
}

at::Tensor check_weight_opt(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t ignore_index) {
  at::Tensor weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor = at::ones(self.size(1), self.options());
  if (weight.defined()) {
    weight_tensor = npu_utils::format_contiguous(weight);
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    at::Tensor zero = at::zeros(1, self.options());
    calcu_op_util::AclrtMemcpyAsync(
        {weight_tensor, ignore_index},
        weight_tensor.itemsize(),
        {zero, 0},
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }
  return weight_tensor;
}

std::tuple<at::Tensor&, at::Tensor&> nll_loss2d_forward_out_nocheck(
    at::Tensor& result,
    at::Tensor& total_weight,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight_tensor,
    int64_t reduction,
    int64_t ignore_index) {
  auto reduction_str = op_plugin::utils::get_reduction_str(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(target)
      .Input(weight_tensor)
      .Attr("reduction", reduction_str)
      .Attr("ignore_index", ignore_index)
      .Output(result)
      .Output(total_weight)
      .Run();

  op_plugin::npu_reshape_out(result, result.sizes(), true, result);
  return std::tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> nll_loss2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& result,
    at::Tensor& total_weight) {
  at::Tensor weight_tensor = check_weight_opt(self, weight_opt, ignore_index);
  auto output_sizes = nll_loss2d_npu_output_size(self, target, reduction, ignore_index);

  npu_preparation::CheckOut(
      {self, target, weight_tensor},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      std::get<0>(output_sizes));

  npu_preparation::CheckOut(
      {self, target, weight_tensor},
      total_weight,
      ACL_FORMAT_ND,
      self.scalar_type(),
      std::get<1>(output_sizes));

  bool result_match = npu_utils::check_match(&result);
  bool total_weight_match = npu_utils::check_match(&total_weight);

  if (!(result_match && total_weight_match)) {
    at::Tensor contiguous_result = result_match ? result : npu_utils::format_contiguous(result);
    at::Tensor contiguous_total_weight = total_weight_match ?
        total_weight : npu_utils::format_contiguous(total_weight);

    nll_loss2d_forward_out_nocheck(contiguous_result, contiguous_total_weight, self, target, weight_tensor, reduction,
        ignore_index);

    if (!result_match) {
      npu_utils::format_fresh_view(result, contiguous_result);
    }
    if (!total_weight_match) {
      npu_utils::format_fresh_view(total_weight, contiguous_total_weight);
    }
  } else {
    nll_loss2d_forward_out_nocheck(result, total_weight, self, target, weight_tensor, reduction, ignore_index);
  }

  return std::tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}

std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // Check Target Dtype
  auto scalar_type = target.scalar_type();
  TORCH_CHECK(scalar_type == at::kLong || scalar_type == at::kInt,
      "Expected object of scalar type ", at::kLong, " or ", at::kInt,
      " but got scalar type ", scalar_type, " for argument 'target' in call to nll_loss2d_forward");
  at::Tensor target_cast = (scalar_type == at::kLong) ? op_plugin::npu_dtype_cast(target, at::kInt) : target;

  auto self_input = self.contiguous();
  self_input = at_npu::native::NPUNativeFunctions::npu_format_cast(self_input, ACL_FORMAT_ND);
  self_input = self_input.permute({0, 2, 3, 1});
  self_input = self_input.reshape({-1, self.size(1)});

  auto target_input = target_cast.contiguous();
  target_input = target_cast.reshape({-1});

  auto output_sizes = nll_loss2d_npu_output_size(self_input, target, reduction, ignore_index);

  at::Tensor result = npu_preparation::apply_tensor(self_input, std::get<0>(output_sizes));
  at::Tensor total_weight = npu_preparation::apply_tensor(self_input, std::get<1>(output_sizes));

  at::Tensor weight_tensor = check_weight_opt(self, weight_opt, ignore_index);
  nll_loss2d_forward_out_nocheck(result, total_weight, self_input, target_input, weight_tensor, reduction,
      ignore_index);
  if (reduction == at::Reduction::None) {
    result.resize_({self.size(0), self.size(2), self.size(3)});
  }

  return std::tuple<at::Tensor, at::Tensor>(result, total_weight);
}
} // namespace op_plugin
