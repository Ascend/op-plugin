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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/AdvancedIndex.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor index_high_dims_op_api(const at::Tensor& self, std::vector<at::Tensor> indices)
{
    std::vector<at::Tensor> all_defined_indices;
    at::SmallVector<int64_t, op_infer::N> zeroSize = {0};
    at::Tensor emptyTensor = npu_preparation::apply_tensor_without_format(self, zeroSize);
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i].defined()) {
        all_defined_indices.emplace_back(indices[i]);
        continue;
        }
        all_defined_indices.emplace_back(emptyTensor);
    }
    auto output_size = op_infer::index_npu_output_size(self, indices);
    auto result = npu_preparation::apply_tensor_without_format(self, output_size);
    at::TensorList indices_tensor_list = all_defined_indices;

    EXEC_NPU_CMD(aclnnIndex, self, indices_tensor_list, result);
    return result;
}

at::Tensor index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig)
{
    DO_COMPATIBILITY(aclnnIndex, acl_op::index(self, orig));
    if (self.device().type() == at::kCPU) {
        return at::native::index(self, orig);
    }
    bool needCast = op_plugin::AdvanceIndex::checkIndexTensorTypes(orig);
    auto indices = op_plugin::AdvanceIndex::npu_expand_tensors(self, orig, needCast, true);
    return index_high_dims_op_api(self, indices);
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor index_high_dims_op_api(const at::Tensor& self, std::vector<at::Tensor> indices)
{
    std::vector<at::Tensor> all_defined_indices;
    at::SmallVector<int64_t, op_infer::N> zeroSize = {0};
    at::Tensor emptyTensor = npu_preparation::apply_tensor_without_format(self, zeroSize);
    for (uint64_t i = 0; i < indices.size(); i++) {
        if (indices[i].defined()) {
        all_defined_indices.emplace_back(indices[i]);
        continue;
        }
        all_defined_indices.emplace_back(emptyTensor);
    }
    auto output_size = op_infer::index_npu_output_size(self, indices);
    auto result = npu_preparation::apply_tensor_without_format(self, output_size);
    at::TensorList indices_tensor_list = all_defined_indices;

    EXEC_NPU_CMD(aclnnIndex, self, indices_tensor_list, result);
    return result;
}

at::Tensor index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig)
{
    DO_COMPATIBILITY(aclnnIndex, acl_op::index(self, orig));
    bool needCast = op_plugin::AdvanceIndex::checkIndexTensorTypes(orig);
    auto indices = op_plugin::AdvanceIndex::npu_expand_tensors(self, orig, needCast, true);
    return index_high_dims_op_api(self, indices);
}
#endif
}
