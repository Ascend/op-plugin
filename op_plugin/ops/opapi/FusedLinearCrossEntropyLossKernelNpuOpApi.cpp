// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.

#include <ATen/TensorSubclassLikeUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_linear_online_max_sum(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & target,
    int64_t vocab_start_index,
    int64_t vocab_end_index,
    bool return_logits)
{
    auto output_size_0 = {input.size(0)};
    auto output_size_1 = {(input.size(0)+7)/8};
    auto output_dtype_0 = at::kFloat;
    auto output_dtype_1 = at::kByte;
    auto output_dtype_2 = target.scalar_type();
    auto output_dtype_3 = input.scalar_type();

    at::Tensor vocab_parallel_logits;
    if (return_logits) {
        auto output_size_2 = c10::SmallVector<int64_t, op_infer::SIZE>{input.size(0), weight.size(0)};
        vocab_parallel_logits = npu_preparation::apply_tensor_without_format(output_size_2, input.options().dtype(output_dtype_3));
    } else {
        vocab_parallel_logits = return_logits ? vocab_parallel_logits : at::Tensor();
    }

    at::Tensor logits_max = npu_preparation::apply_tensor_without_format(output_size_0, input.options().dtype(output_dtype_0));
    at::Tensor sum_exp_logits = npu_preparation::apply_tensor_without_format(output_size_0, input.options().dtype(output_dtype_0));
    at::Tensor predicted_logits = npu_preparation::apply_tensor_without_format(output_size_0, input.options().dtype(output_dtype_0));
    at::Tensor target_mask = npu_preparation::apply_tensor_without_format(output_size_1, input.options().dtype(output_dtype_1));
    at::Tensor masked_target = npu_preparation::apply_tensor_without_format(output_size_0, input.options().dtype(output_dtype_2));

    EXEC_NPU_CMD(aclnnFusedLinearOnlineMaxSum, input, weight, target, vocab_start_index, vocab_end_index, logits_max, sum_exp_logits, predicted_logits, target_mask, masked_target, vocab_parallel_logits);
    return std::make_tuple(std::move(logits_max), std::move(sum_exp_logits), std::move(predicted_logits), std::move(target_mask), std::move(masked_target), std::move(vocab_parallel_logits));
}

::std::tuple<at::Tensor, at::Tensor> fused_cross_entropy_loss_with_max_sum(
    const at::Tensor & logits_max,
    const at::Tensor & sum_exp_logits,
    const at::Tensor & predicted_logits,
    c10::optional<double> label_smoothing,
    const c10::optional<at::Tensor> & input,
    const c10::optional<at::Tensor> & weight,
    const c10::optional<at::Tensor> & vocab_parallel_logits)
{
    auto label_smoothing_value = label_smoothing.value_or(0.0);
    auto output_size_0 = logits_max.sizes();
    auto output_dtype_0 = at::kFloat;

    at::Tensor softmax;
    if (vocab_parallel_logits.has_value() && vocab_parallel_logits.value().defined()) {
        auto output_size_1 = vocab_parallel_logits.value().sizes();
        softmax = npu_preparation::apply_tensor_without_format(output_size_1, logits_max.options().dtype(output_dtype_0));
    } else {
        softmax = at::Tensor();
    }

    at::Tensor loss = npu_preparation::apply_tensor_without_format(output_size_0, logits_max.options().dtype(output_dtype_0));

    EXEC_NPU_CMD(aclnnFusedCrossEntropyLossWithMaxSum, logits_max, sum_exp_logits, predicted_logits, label_smoothing_value, input, weight, vocab_parallel_logits, loss, softmax);
    return std::make_tuple(std::move(loss), std::move(softmax));
}

::std::tuple<at::Tensor, at::Tensor> fused_linear_cross_entropy_loss_with_max_sum_grad(
    const at::Tensor & grad,
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor &  target_mask,
    const at::Tensor & masked_target,
    double label_smoothing,
    const c10::optional<at::Tensor> & logits_max,
    const c10::optional<at::Tensor> & sum_exp_logits,
    const c10::optional<at::Tensor> & softmax)
{
    auto output_size_0 = {input.size(0), input.size(1)};
    auto output_size_1 = {weight.size(0), weight.size(1)};
    auto output_dtype_0 = input.scalar_type();
    at::Tensor input_grad = npu_preparation::apply_tensor_without_format(output_size_0, grad.options().dtype(output_dtype_0));
    at::Tensor weight_grad = npu_preparation::apply_tensor_without_format(output_size_1, grad.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnFusedLinearCrossEntropyLossGrad, grad, input, weight, target_mask, masked_target, label_smoothing, logits_max, sum_exp_logits, softmax, input_grad, weight_grad);
    return std::make_tuple(std::move(input_grad), std::move(weight_grad));
}
}