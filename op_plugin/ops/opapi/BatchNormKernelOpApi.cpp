// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/AdvancedIndex.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Both helpers below only speed up the write-back of the inference statistics; the
// values they produce are the ones the plain copy_ / rsqrt path produces.

// save_mean is just running_mean. For equal dtype, contiguous ND tensors copy_ already
// ends up in copy_d2d_by_memcpy, so calling it directly issues the very same
// device-to-device copy task while skipping copy_'s dtype / format / contiguity analysis
// chain. Direct use of this helper matches ReshapeKernel in both backends.
void write_inference_mean(const at::Tensor& mean, at::Tensor& save_mean)
{
    if (save_mean.scalar_type() == mean.scalar_type() && save_mean.sizes().equals(mean.sizes()) &&
        save_mean.device() == mean.device() &&
        npu_preparation::get_tensor_npu_format(save_mean) == ACL_FORMAT_ND &&
        npu_preparation::get_tensor_npu_format(mean) == ACL_FORMAT_ND &&
        save_mean.is_contiguous() && mean.is_contiguous() &&
        at_npu::native::NpuUtils::check_match(&save_mean) && at_npu::native::NpuUtils::check_match(&mean) &&
        at_npu::native::get_overlap_status(save_mean, mean) == at_npu::native::MemOverlapStatus::NO) {
        at_npu::native::copy_d2d_by_memcpy(save_mean, mean, save_mean.numel());
        return;
    }
    save_mean.copy_(mean);
}

// aclnnBatchNorm leaves save_mean / save_invstd undefined for training=false, so the
// inference statistics have to be produced here. When save_invstd can be written
// directly, the eps-add goes straight into it and the rsqrt runs in place: still the
// same two device ops, but without the temporary tensor and its dispatcher round trips.
void write_inference_invstd(const at::Tensor& variance, double eps, at::Tensor& save_invstd)
{
    static const bool fused_invstd_available = check_aclnn_kernel_available("aclnnAdds") &&
        check_aclnn_kernel_available("aclnnInplaceRsqrt");
    if (save_invstd.scalar_type() == variance.scalar_type() && save_invstd.sizes().equals(variance.sizes()) &&
        npu_preparation::get_tensor_npu_format(save_invstd) == ACL_FORMAT_ND &&
        npu_preparation::get_tensor_npu_format(variance) == ACL_FORMAT_ND &&
        at_npu::native::NpuUtils::check_match(&save_invstd)) {
        // save_invstd is written before variance is read, so memory shared with running_var
        // would corrupt the running statistics; those inputs keep the temporary tensor.
        if (fused_invstd_available &&
            at_npu::native::get_overlap_status(save_invstd, variance) == at_npu::native::MemOverlapStatus::NO) {
            const at::Scalar eps_scalar = eps;
            const at::Scalar alpha = 1;
            EXEC_NPU_CMD(aclnnAdds, variance, eps_scalar, alpha, save_invstd);
            EXEC_NPU_CMD(aclnnInplaceRsqrt, save_invstd);
            return;
        }
        at::rsqrt_out(save_invstd, variance + eps);
        return;
    }
    save_invstd.copy_(at::rsqrt(variance + eps));
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps,
    at::Tensor& out, at::Tensor& save_mean, at::Tensor& save_invstd) {
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm_out(
        input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd));
    npu_preparation::check_tensor({input}, out, out.scalar_type(), out.sizes());
    npu_preparation::check_tensor({input}, save_mean, save_mean.scalar_type(), save_mean.sizes());
    npu_preparation::check_tensor({input}, save_invstd, save_invstd.scalar_type(), save_invstd.sizes());
    EXEC_NPU_CMD(aclnnBatchNorm, input, weight, bias, running_mean, running_var,
                 training, momentum, eps, out, save_mean, save_invstd);
    if (!training && running_mean.has_value() && running_mean->defined() &&
        running_var.has_value() && running_var->defined()) {
        write_inference_mean(running_mean.value(), save_mean);
        auto variance = running_var.value();
        if (variance.scalar_type() == at::kHalf || variance.scalar_type() == at::kBFloat16) {
            variance = variance.to(at::kFloat);
        }
        TORCH_CHECK(op_plugin::AdvanceIndex::is_expandable_to(variance.sizes(), save_invstd.sizes()),
            "shape mismatch: running_var of shape ", variance.sizes(),
            " cannot be broadcast to save_invstd of shape ", save_invstd.sizes(),
            OPS_ERROR(ErrCode::PARAM));
        write_inference_invstd(variance, eps, save_invstd);
    }
    return std::tie(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps));
    at::Tensor out = npu_preparation::apply_tensor_without_format(input.sizes(), input.options());
    auto stats_options = input.options().dtype(training ? at::kFloat : input.scalar_type());
    at::Tensor save_mean = npu_preparation::apply_tensor_without_format({input.size(1)}, stats_options);
    at::Tensor save_invstd = npu_preparation::apply_tensor_without_format({input.size(1)}, stats_options);
    op_api::native_batch_norm_out(input, weight, bias, running_mean, running_var,
                                  training, momentum, eps, out, save_mean, save_invstd);
    return std::make_tuple(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(const at::Tensor &input,
    const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias, at::Tensor &running_mean,
    at::Tensor &running_var, bool training, double momentum, double eps) {
    return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, bool training, double momentum, double eps) {
    TORCH_CHECK(training, "_native_batch_norm_legit.no_stats requires training=True", OPS_ERROR(ErrCode::PARAM));
    return at::native_batch_norm(
        input, weight, bias, c10::nullopt, c10::nullopt, training, momentum, eps);
}

} // namespace op_api
