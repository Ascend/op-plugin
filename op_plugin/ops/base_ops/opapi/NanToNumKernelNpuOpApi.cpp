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

namespace {
const float DOUBLE_MAX_VALUE = 1.7976931348623157e+308;
const float DOUBLE_MIN_VALUE = -1.7976931348623157e+308;
const float FLOAT32_MAX_VALUE = 3.4028235e+38;
const float FLOAT32_MIN_VALUE = -3.4028235e+38;
const float FLOAT16_MAX_VALUE = 65504.0;
const float FLOAT16_MIN_VALUE = -65504.0;
const float BFLOAT16_MAX_VALUE = 3.3895314e+38;
const float BFLOAT16_MIN_VALUE = -3.3895314e+38;
const float DEFAULT_NAN = 0.0;

std::tuple<float, float> get_posinf_and_neginf(at::ScalarType self_dtype,
                                               c10::optional<double> posinf,
                                               c10::optional<double> neginf)
{
    float new_posinf;
    float new_neginf;
    bool posinf_has_value = posinf.has_value();
    bool neginf_has_value = neginf.has_value();
    if (posinf_has_value && neginf_has_value) {
        new_posinf = posinf.value();
        new_neginf = neginf.value();
    } else {
        switch (self_dtype) {
            case at::ScalarType::Double:
                new_posinf = posinf_has_value ? posinf.value() : DOUBLE_MAX_VALUE;
                new_neginf = neginf_has_value ? neginf.value() : DOUBLE_MIN_VALUE;
                break;
            case at::ScalarType::Float:
                new_posinf = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
                new_neginf = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
                break;
            case at::ScalarType::Half:
                new_posinf = posinf_has_value ? posinf.value() : FLOAT16_MAX_VALUE;
                new_neginf = neginf_has_value ? neginf.value() : FLOAT16_MIN_VALUE;
                break;
            case at::ScalarType::BFloat16:
                new_posinf = posinf_has_value ? posinf.value() : BFLOAT16_MAX_VALUE;
                new_neginf = neginf_has_value ? neginf.value() : BFLOAT16_MIN_VALUE;
                break;
            default:
                new_posinf = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
                new_neginf = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
                break;
        }
    }
    return std::tie(new_posinf, new_neginf);
}
}

at::Tensor& nan_to_num_out(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> pos_inf,
                           c10::optional<double> neg_inf, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnNanToNum, acl_op::nan_to_num_out(self, nan, pos_inf, neg_inf, result));

    at_npu::native::OpPreparation::check_tensor({self}, result, self.scalar_type(), self.sizes());
    float new_nan = nan.has_value() ? nan.value() : DEFAULT_NAN;
    auto new_posinf_neginf = get_posinf_and_neginf(self.scalar_type(), pos_inf, neg_inf);
    float new_posinf = std::get<0>(new_posinf_neginf);
    float new_neginf = std::get<1>(new_posinf_neginf);
    EXEC_NPU_CMD(aclnnNanToNum, self, new_nan, new_posinf, new_neginf, result);
    return result;
}

at::Tensor nan_to_num(const at::Tensor& self, c10::optional<double> nan, c10::optional<double> pos_inf,
                      c10::optional<double> neg_inf)
{
    DO_COMPATIBILITY(aclnnNanToNum, acl_op::nan_to_num(self, nan, pos_inf, neg_inf));
    // construct the output tensor of the NPU
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());
    float new_nan = nan.has_value() ? nan.value() : DEFAULT_NAN;
    auto new_posinf_neginf = get_posinf_and_neginf(self.scalar_type(), pos_inf, neg_inf);
    float new_posinf = std::get<0>(new_posinf_neginf);
    float new_neginf = std::get<1>(new_posinf_neginf);

    EXEC_NPU_CMD(aclnnNanToNum, self, new_nan, new_posinf, new_neginf, result);
    return result;
}

at::Tensor& nan_to_num_(at::Tensor& self, c10::optional<double> nan, c10::optional<double> pos_inf,
                        c10::optional<double> neg_inf)
{
    DO_COMPATIBILITY(aclnnInplaceNanToNum, acl_op::nan_to_num_(self, nan, pos_inf, neg_inf));

    float new_nan = nan.has_value() ? nan.value() : DEFAULT_NAN;
    auto new_posinf_neginf = get_posinf_and_neginf(self.scalar_type(), pos_inf, neg_inf);
    float new_posinf = std::get<0>(new_posinf_neginf);
    float new_neginf = std::get<1>(new_posinf_neginf);

    EXEC_NPU_CMD(aclnnInplaceNanToNum, self, new_nan, new_posinf, new_neginf);
    return self;
}
} // namespace op_api
