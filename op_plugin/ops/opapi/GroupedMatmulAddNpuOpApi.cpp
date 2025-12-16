// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& npu_grouped_matmul_add_(at::Tensor& self, const at::Tensor& x, const at::Tensor& weight,
                                    const at::Tensor& group_list, bool transpose_x, bool transpose_weight,
                                    int64_t group_type, c10::optional<int64_t> group_list_type)
{
    static const bool has_grouped_matmul_add_v2 = check_aclnn_kernel_available("aclnnGroupedMatmulAddV2");
    int64_t group_list_type_value = group_list_type.value_or(0);
    static const bool checkSoc = c10_npu::IsAclnnOnly();
    if (!has_grouped_matmul_add_v2 || !checkSoc) {
        EXEC_NPU_CMD(aclnnGroupedMatmulAdd, x, weight, group_list, self, transpose_x, transpose_weight,
                     group_type);
        return self;
    }
    EXEC_NPU_CMD(aclnnGroupedMatmulAddV2, x, weight, group_list, self, transpose_x, transpose_weight,
                 group_type, group_list_type_value);

    return self;
}

at::Tensor npu_grouped_matmul_add(const at::Tensor& self, const at::Tensor& x, const at::Tensor& weight,
                                  const at::Tensor& group_list, bool transpose_x, bool transpose_weight,
                                  int64_t group_type, c10::optional<int64_t> group_list_type)
{
    static const bool has_grouped_matmul_add_v2 = check_aclnn_kernel_available("aclnnGroupedMatmulAddV2");
    int64_t group_list_type_value = group_list_type.value_or(0);
    static const bool checkSoc = c10_npu::IsAclnnOnly();
    if (!has_grouped_matmul_add_v2 || !checkSoc) {
        EXEC_NPU_CMD(aclnnGroupedMatmulAdd, x, weight, group_list, self, transpose_x, transpose_weight, group_type);
        return self.clone();
    }
    EXEC_NPU_CMD(aclnnGroupedMatmulAddV2, x, weight, group_list, self, transpose_x, transpose_weight, group_type,
                 group_list_type_value);

    return self.clone();
}

} // namespace op_api
