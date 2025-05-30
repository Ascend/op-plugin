// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline int64_t _masked_softmax_mask_type(const at::Tensor& self_, const at::Tensor& mask_, const c10::optional<int64_t>& mask_type_)
{
    TORCH_CHECK(mask_.scalar_type() == at::ScalarType::Bool, "Mask should be a boolean tensor", OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mask_type_.has_value(), "Mask Type should be defined" + OPS_ERROR(ErrCode::PARAM));
    int64_t mask_type = mask_type_.value();
    TORCH_CHECK((mask_type == 0) || (mask_type == 1) || (mask_type == 2),
        "Mask Type should be 0 (src_mask), 1 (src_key_padding_mask), or 2 (default_mask)", OPS_ERROR(ErrCode::VALUE));
    if (mask_type == 2) {
        TORCH_CHECK(mask_.sizes() == self_.sizes(), "Mask shape should match input. mask: ", mask_.sizes(), " input: ", self_.sizes(), OPS_ERROR(ErrCode::PARAM));
    }
    return mask_type;
}

inline bool _masked_softmax_fallback_condition(
    const at::Tensor& self,
    const c10::optional<int64_t> dim,
    const c10::optional<int64_t> mask_type_)
{
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnScaledMaskedSoftmax");
    if (!is_aclnn_kernel_available) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::_masked_softmax is currently not supported on the NPU backend."
            "Now this operator will fallback to run on the CPU and may have performance implications. "
            "Please try to update your CANN version.");
        return true;
    }
    bool dim_use_npu = (!dim.has_value()) || (dim.value() == -1) || (dim.value() == self.dim() - 1);
    if ((self.dim() != 4) || (!dim_use_npu) || (!mask_type_.has_value()) || (mask_type_ != 2)) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::_masked_softmax need to be met the following conditions to "
            "run on the NPU backend: 1. dim = None/dim = -1/dim = input.dim()-1; 2. mask_type = 2; 3. input.dim() = 4. "
            "Now this operator will fallback to run on the CPU and may have performance implications. "
            "Please check your params.");
        return true;
    }
    return false;
}

at::Tensor _masked_softmax(
    const at::Tensor& self_,
    const at::Tensor& mask_,
    const c10::optional<int64_t> dim_,
    const c10::optional<int64_t> mask_type_)
{
    if (_masked_softmax_fallback_condition(self_, dim_, mask_type_)) {
        at::Tensor self_cpu = self_.cpu();
        at::Tensor mask_cpu = mask_.cpu();
        at::Tensor result = at::_masked_softmax(self_cpu, mask_cpu, dim_, mask_type_);
        return result.to(self_.device());
    }
    double scale = 1.0;
    bool fixed_triu_mask = false;
    at::Tensor self = self_.dim() == 0 ? self_.view(1) : self_;
    at::Tensor mask = mask_.dim() == 0 ? mask_.view(1) : mask_;
    int64_t mask_type = _masked_softmax_mask_type(self_, mask_, mask_type_);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options());
    EXEC_NPU_CMD(aclnnScaledMaskedSoftmax, self, mask, scale, fixed_triu_mask, result);
    return result;
}

}  // namespace op_api
