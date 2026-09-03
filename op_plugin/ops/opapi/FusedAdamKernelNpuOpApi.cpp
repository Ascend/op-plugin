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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

#include <ATen/ops/_fused_adam.h>
#include <ATen/native/ForeachUtils.h>

#include <algorithm>
#include <vector>

namespace op_api {
namespace {
void fused_adam_cpu_fallback(
	at::TensorList self,
	at::TensorList grads,
	at::TensorList exp_avgs,
	at::TensorList exp_avg_sqs,
	at::TensorList max_exp_avg_sqs,
	at::TensorList state_steps,
	const double lr,
	const double beta1,
	const double beta2,
	const double weight_decay,
	const double eps,
	const bool amsgrad,
	const bool maximize,
	const c10::optional<at::Tensor>& grad_scale,
	const c10::optional<at::Tensor>& found_inf)
{
	std::vector<at::Tensor> self_cpu;
	std::vector<at::Tensor> grads_cpu;
	std::vector<at::Tensor> exp_avgs_cpu;
	std::vector<at::Tensor> exp_avg_sqs_cpu;
	std::vector<at::Tensor> max_exp_avg_sqs_cpu;
	std::vector<at::Tensor> state_steps_cpu;
	self_cpu.reserve(self.size());
	grads_cpu.reserve(grads.size());
	exp_avgs_cpu.reserve(exp_avgs.size());
	exp_avg_sqs_cpu.reserve(exp_avg_sqs.size());
	max_exp_avg_sqs_cpu.reserve(max_exp_avg_sqs.size());
	state_steps_cpu.reserve(state_steps.size());

	for (const auto& tensor : self) {
		self_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : grads) {
		grads_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : exp_avgs) {
		exp_avgs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : exp_avg_sqs) {
		exp_avg_sqs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : max_exp_avg_sqs) {
		max_exp_avg_sqs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : state_steps) {
		state_steps_cpu.emplace_back(tensor.cpu());
	}

	c10::optional<at::Tensor> grad_scale_cpu =
		grad_scale.has_value() ? c10::optional<at::Tensor>(grad_scale->cpu()) : c10::nullopt;
	c10::optional<at::Tensor> found_inf_cpu =
		found_inf.has_value() ? c10::optional<at::Tensor>(found_inf->cpu()) : c10::nullopt;

	at::_fused_adam_(
		self_cpu,
		grads_cpu,
		exp_avgs_cpu,
		exp_avg_sqs_cpu,
		max_exp_avg_sqs_cpu,
		state_steps_cpu,
		lr,
		beta1,
		beta2,
		weight_decay,
		eps,
		amsgrad,
		maximize,
		grad_scale_cpu,
		found_inf_cpu);

	for (size_t i = 0; i < self.size(); ++i) {
		self[i].copy_(self_cpu[i]);
		exp_avgs[i].copy_(exp_avgs_cpu[i]);
		exp_avg_sqs[i].copy_(exp_avg_sqs_cpu[i]);
		if (amsgrad) {
			max_exp_avg_sqs[i].copy_(max_exp_avg_sqs_cpu[i]);
		}
	}
}

void _split_and_exec_npu_cmd_fused_adam(
	at::TensorList self,
	at::TensorList grads,
	at::TensorList exp_avgs,
	at::TensorList exp_avg_sqs,
	at::TensorList max_exp_avg_sqs,
	at::TensorList state_steps,
	const at::Tensor& grad_scale_tensor,
	const at::Tensor& found_inf_tensor,
	const double lr,
	const double beta1,
	const double beta2,
	const double weight_decay,
	const double eps,
	const bool amsgrad,
	const bool maximize)
{
	constexpr size_t max_tensor_count = 512;
	if (self.empty()) {
		return;
	}

	for (size_t offset = 0; offset < self.size(); offset += max_tensor_count) {
		size_t current_count = std::min(max_tensor_count, self.size() - offset);
		at::TensorList self_batch(self.data() + offset, current_count);
		at::TensorList grads_batch(grads.data() + offset, current_count);
		at::TensorList exp_avgs_batch(exp_avgs.data() + offset, current_count);
		at::TensorList exp_avg_sqs_batch(exp_avg_sqs.data() + offset, current_count);
		at::TensorList max_exp_avg_sqs_batch(max_exp_avg_sqs.data() + offset, current_count);
		at::TensorList state_steps_batch(state_steps.data() + offset, current_count);

		EXEC_NPU_CMD(
			aclnnFusedAdam,
			self_batch,
			grads_batch,
			exp_avgs_batch,
			exp_avg_sqs_batch,
			max_exp_avg_sqs_batch,
			state_steps_batch,
			grad_scale_tensor,
			found_inf_tensor,
			lr,
			beta1,
			beta2,
			weight_decay,
			eps,
			amsgrad,
			maximize);
	}
}
} // namespace

void _fused_adam_(
	at::TensorList self,
	at::TensorList grads,
	at::TensorList exp_avgs,
	at::TensorList exp_avg_sqs,
	at::TensorList max_exp_avg_sqs,
	at::TensorList state_steps,
	const double lr,
	const double beta1,
	const double beta2,
	const double weight_decay,
	const double eps,
	const bool amsgrad,
	const bool maximize,
	const c10::optional<at::Tensor>& grad_scale,
	const c10::optional<at::Tensor>& found_inf)
{
	if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
		TORCH_NPU_WARN(
			"CAUTION: The operator 'aten::_fused_adam_' is not currently supported "
			"on the NPU backend and will fall back to run on the CPU."
			" This may have performance implications.");
		fused_adam_cpu_fallback(
			self,
			grads,
			exp_avgs,
			exp_avg_sqs,
			max_exp_avg_sqs,
			state_steps,
			lr,
			beta1,
			beta2,
			weight_decay,
			eps,
			amsgrad,
			maximize,
			grad_scale,
			found_inf);
		return;
	}

	bool is_same_size = (self.size() == grads.size() &&
					   self.size() == exp_avgs.size() &&
					   self.size() == exp_avg_sqs.size() &&
					   self.size() == state_steps.size() &&
					   (max_exp_avg_sqs.size() == 0 ||
					   self.size() == max_exp_avg_sqs.size()));
	if (!is_same_size) {
		TORCH_CHECK(false, "the size of tensor list should be same.");
	}

	std::vector<at::Tensor> state_steps_adjusted;
	state_steps_adjusted.reserve(state_steps.size());
	for (size_t i = 0; i < state_steps.size(); i++) {
		state_steps_adjusted.emplace_back(state_steps[i].sub(1));
	}
	std::vector<at::Tensor> max_exp_avg_sqs_adjusted;
	if (max_exp_avg_sqs.empty()) {
		max_exp_avg_sqs_adjusted.reserve(self.size());
		for (const at::Tensor& tensor : self) {
			max_exp_avg_sqs_adjusted.emplace_back(at::zeros_like(tensor));
		}
	}
	at::TensorList state_steps_list(state_steps_adjusted);
	at::TensorList max_exp_avg_sqs_list = max_exp_avg_sqs.empty()
		? at::TensorList(max_exp_avg_sqs_adjusted)
		: max_exp_avg_sqs;
	const at::Tensor grad_scale_tensor = grad_scale.value_or(at::Tensor());
	const at::Tensor found_inf_tensor = found_inf.value_or(at::Tensor());

	_split_and_exec_npu_cmd_fused_adam(
		self,
		grads,
		exp_avgs,
		exp_avg_sqs,
		max_exp_avg_sqs_list,
		state_steps_list,
		grad_scale_tensor,
		found_inf_tensor,
		lr,
		beta1,
		beta2,
		weight_decay,
		eps,
		amsgrad,
		maximize);
}
}  // namespace op_api
