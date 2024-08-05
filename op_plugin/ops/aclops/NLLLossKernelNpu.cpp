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
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
void nll_loss_forward_check(const at::Tensor& self, const at::Tensor& target) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D"
      + OPS_ERROR(ErrCode::PARAM));
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported"
      + OPS_ERROR(ErrCode::PARAM));
  auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || self.size(0) == target.size(0),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")"
      + OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&> nll_loss_forward_out_nocheck(
    at::Tensor& result,
    at::Tensor& total_weight,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = npu_utils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    at::Tensor zero = at::zeros(1, self.options());
    calcu_op_util::AclrtMemcpyAsync({weight_tensor, ignore_index}, weight_tensor.itemsize(),
        {zero, 0}, weight_tensor.itemsize(), ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  string reduction_str = op_plugin::utils::get_reduction_str(reduction);

  auto scalar_type = target.scalar_type();
  TORCH_CHECK((scalar_type == at::kLong || scalar_type == at::kInt),
      "Expected object of scalar type ", at::kLong, " or ", at::kInt,
      " but got scalar type ", scalar_type, " for argument 'target' in call to nll_loss_forward"
      + OPS_ERROR(ErrCode::TYPE));
  at::Tensor target_cast = (scalar_type == at::kLong) ? at_npu::native::custom_ops::npu_dtype_cast(target, at::kInt) : target;

  at_npu::native::OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(target_cast)
      .Input(weight_tensor)
      .Output(result)
      .Output(total_weight)
      .Attr("reduction", reduction_str)
      .Attr("ignore_index", ignore_index)
      .Run();

  return std::tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> nll_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& result,
    at::Tensor& total_weight) {
  nll_loss_forward_check(self, target);
  at::Tensor self_cp = self.dim() == 1 ? self.unsqueeze(0) : self;
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = npu_utils::format_contiguous(weight);
  } else {
    auto options = self_cp.options();
    weight_tensor = acl_op::ones(
        self_cp.size(1),
        c10::optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  }

  c10::SmallVector<int64_t, SIZE> output_size = {};
  if (reduction == at::Reduction::None) {
    output_size = {self_cp.size(0)};
  }

  npu_preparation::CheckOut(
      {self_cp, target, weight_tensor},
      result,
      ACL_FORMAT_ND,
      self_cp.scalar_type(),
      output_size);

  npu_preparation::CheckOut(
      {self_cp, target, weight_tensor},
      total_weight,
      ACL_FORMAT_ND,
      self_cp.scalar_type(),
      {});

  bool result_match = npu_utils::check_match(&result);
  bool total_weight_match = npu_utils::check_match(&total_weight);
  if (!(result_match && total_weight_match)) {
    at::Tensor contiguous_result = result_match ? result : npu_utils::format_contiguous(result);
    at::Tensor contiguous_total_weight =
        total_weight_match ? total_weight : npu_utils::format_contiguous(total_weight);

    nll_loss_forward_out_nocheck(contiguous_result, contiguous_total_weight, self_cp,
        target, weight, reduction, ignore_index);

    if (!result_match) {
      npu_utils::format_fresh_view(result, contiguous_result);
    }
    if (!total_weight_match) {
      npu_utils::format_fresh_view(total_weight, contiguous_total_weight);
    }
  } else {
    nll_loss_forward_out_nocheck(result, total_weight, self_cp, target, weight, reduction, ignore_index);
  }

  if (self.dim() == 1 && reduction == at::Reduction::None) {
    result.squeeze_(0);
  }
  return std::tie(result, total_weight);
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // ND case
  nll_loss_forward_check(self, target);
  at::Tensor self_cp = self.dim() == 1 ? self.unsqueeze(0) : self;
  c10::SmallVector<int64_t, SIZE> output_size = {};
  c10::SmallVector<int64_t, SIZE> total_weight_size = {};
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  if (reduction == at::Reduction::None) {
    output_size = {self_cp.size(0)};
  }

  // Special output, output' dim is <= 1 fixedly
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      self_cp, output_size, ACL_FORMAT_ND);
  at::Tensor total_weight = npu_preparation::apply_tensor_with_format(
      self_cp, total_weight_size, ACL_FORMAT_ND);

  nll_loss_forward_out_nocheck(result, total_weight, self_cp,
      target, weight, reduction, ignore_index);
  if (self.dim() == 1 && reduction == at::Reduction::None) {
    result.squeeze_(0);
  }
  return std::tuple<at::Tensor, at::Tensor>(result, total_weight);
}
} // namespace acl_op
