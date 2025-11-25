// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/OpUtils.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_add_quant_gmm_symint(const at::Tensor &self, const at::Tensor &x1, const at::Tensor &x2,
                                    const at::Tensor &x2_scale, const at::Tensor &group_list,
                                    const c10::optional<at::Tensor> &x1_scale, c10::optional<int64_t> group_list_type,
                                    c10::OptionalArrayRef<c10::SymInt> group_sizes, c10::optional<int64_t> x1_dtype,
                                    c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1_scale_dtype,
                                    c10::optional<int64_t> x2_scale_dtype)
// func: npu_add_quant_gmm_(Tensor(a!) self, Tensor x1, Tensor x2, Tensor x2_scale, Tensor group_list, *,
// Tensor? x1_scale=None, int? group_list_type=0, int[]? group_sizes=None, int? x1_dtype=None, int? x2_dtype=None,
// int? x1_scale_dtype=None, int? x2_scale_dtype=None) -> Tensor(a!) 对应的非原地npu实现
{
    static const bool is_quant_grouped_matmul_inplace_add_available =
        check_aclnn_kernel_available("aclnnQuantGroupedMatmulInplaceAdd");
    TORCH_CHECK(is_quant_grouped_matmul_inplace_add_available,
                "Get aclnnQuantGroupedMatmulInplaceAdd or aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize failed, "
                "please upgrade CANN.",
                OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(!group_sizes.has_value(), "group_sizes is not supported currently. ", OPS_ERROR(ErrCode::VALUE));

    const at::Tensor &x1_scale_real = x1_scale.value_or(at::Tensor());
    int64_t group_list_type_value = group_list_type.value_or(0);
    int64_t group_size = 0;

    TensorWrapper x1_wrapper = {x1, x1_dtype.has_value() ? c10_npu::GetAclDataType(x1_dtype.value())
                                                         : npu_preparation::convert_to_acl_data_type(x1.scalar_type())};
    TensorWrapper x2_wrapper = {x2, x2_dtype.has_value() ? c10_npu::GetAclDataType(x2_dtype.value())
                                                         : npu_preparation::convert_to_acl_data_type(x2.scalar_type())};
    TensorWrapper x2_scale_wrapper = {
        x2_scale, x2_scale_dtype.has_value() ? c10_npu::GetAclDataType(x2_scale_dtype.value())
                                             : npu_preparation::convert_to_acl_data_type(x2_scale.scalar_type())};
    TensorWrapper x1_scale_wrapper = {
        x1_scale_real,
        x1_scale_dtype.has_value()
            ? c10_npu::GetAclDataType(x1_scale_dtype.value())
            : (x1_scale.has_value() ? npu_preparation::convert_to_acl_data_type(x1_scale_real.scalar_type())
                                    : aclDataType::ACL_FLOAT)};

    auto self_copy = self.clone();
    EXEC_NPU_CMD(aclnnQuantGroupedMatmulInplaceAdd, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper,
                 group_list, self_copy, group_list_type_value, group_size);

    return self_copy;
}

}  // namespace op_api