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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    bool zero_infinity)
{
    DO_COMPATIBILITY(aclnnCtcLoss, acl_op::_ctc_loss(log_probs, targets, input_lengths_list, target_lengths_list, blank,
                                                     zero_infinity));

    int64_t max_length = 0;
    for (auto& i : target_lengths_list) {
        if (i > max_length) {
            max_length = i;
        }
    }

    // calculate the output size
    auto outputSizes = op_infer::ctc_loss_npu_output_size(log_probs, max_length);

    // construct the output tensor of the NPU
    at::Tensor neg_log_likelihood = npu_preparation::apply_tensor_without_format(log_probs, std::get<0>(outputSizes));
    at::Tensor log_alpha = npu_preparation::apply_tensor_without_format(log_probs, std::get<1>(outputSizes));

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnCtcLoss, log_probs, targets, input_lengths_list, target_lengths_list,
                 blank, zero_infinity, neg_log_likelihood, log_alpha);

    return std::tuple<at::Tensor, at::Tensor>(neg_log_likelihood, log_alpha);
}

at::Tensor ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths_list,
    at::IntArrayRef target_lengths_list,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity)
{
    DO_COMPATIBILITY(aclnnCtcLoss, acl_op::ctc_loss(log_probs, targets, input_lengths_list, target_lengths_list, blank,
                                                    reduction, zero_infinity));
    at::Tensor res = std::get<0>(at::_ctc_loss(
        log_probs,
        targets,
        input_lengths_list,
        target_lengths_list,
        blank,
        zero_infinity));

    if (zero_infinity) {
        res = at::where(
            res == at::Scalar(std::numeric_limits<double>::infinity()),
            at::zeros({}, res.options()),
            res);
    }

    if (reduction == at::Reduction::Mean) {
        std::vector<int64_t> target_lengths_vector = target_lengths_list.vec();
        auto target_lengths_tensor = npu_preparation::copy_tensor_host_to_device(
            at::from_blob(target_lengths_vector.data(), {target_lengths_vector.size()}, at::kLong)).clamp_min(1);
        at::Tensor target_lengths_tensor_ = target_lengths_tensor.to(res.dtype());
        return (res / target_lengths_tensor_).mean();
    } else if (reduction == at::Reduction::Sum) {
        return res.sum();
    }

    return res;
}

at::Tensor ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity)
{
    DO_COMPATIBILITY(aclnnCtcLoss, acl_op::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction,
                                                    zero_infinity));
    TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), false), "input_lengths must be integral", OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), false), "target_lengths must be integral", OPS_ERROR(ErrCode::TYPE));

    at::Tensor input_lengths_tensor = input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
    at::Tensor target_lengths_tensor = target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();

    auto input_length_ptr = input_lengths_tensor.data_ptr<int64_t>();
    auto target_length_ptr = target_lengths_tensor.data_ptr<int64_t>();
    TORCH_CHECK(input_length_ptr != nullptr, "input_lengths is null", OPS_ERROR(ErrCode::PTR))
    TORCH_CHECK(target_length_ptr != nullptr, "target_lengths is null", OPS_ERROR(ErrCode::PTR))
    at::IntArrayRef input_lengths_list(input_length_ptr, input_lengths_tensor.numel());
    at::IntArrayRef target_lengths_list(target_length_ptr, target_lengths_tensor.numel());

    return at::ctc_loss(log_probs, targets, input_lengths_list, target_lengths_list, blank, reduction, zero_infinity);
}
}
