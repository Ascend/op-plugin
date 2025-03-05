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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/AdvancedIndex.h"
#include "op_plugin/third_party/acl/inc/op_proto/all_ops.h"

namespace acl_op {
using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;

namespace {
const std::string x_str = "x";
const std::string indexed_sizes_str = "indexed_sizes";
const std::string indexed_strides_str = "indexed_strides";
const std::string aicore_str = "AiCore";

at_npu::native::DynamicInputRegFunc index_func = [](DyNumAndIndex num_and_index,
                                                    std::string op_name) -> ge::OperatorPtr {
    auto ge_op = std::make_shared<ge::op::Index>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_indices(num_and_index.front().first, num_and_index.front().second);
    return ge_op;
};

// Limitations of the aicore branch
bool check_index_aicore(const at::TensorList &indices, const at::IntArrayRef masks)
{
    // indices must have same shape
    for (uint64_t idx = 1; idx < indices.size(); idx++) {
        if (indices[idx].sizes() != indices[idx - 1].sizes()) {
            return false;
        }
    }

    // only supports continuous indices
    for (uint64_t idx = 1; idx < masks.size(); idx++) {
        if (masks[idx] - masks[idx - 1] < 0) {
            return false;
        }
    }
    return true;
}

at::Tensor &index_out_nocheck(const at::Tensor &self, const at::IntArrayRef masks, const at::TensorList &indices,
                              at::Tensor &result, bool is_aicore)
{
    at::IntArrayRef indexed_strides = result.sizes();
    at_npu::native::OpCommand cmd;
    cmd.Name("Index")
        .Input(self, x_str)
        .Input(masks, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT, "", indexed_sizes_str)
        .Input(indexed_strides, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT, "", indexed_strides_str);
    for (uint64_t i = 0; i < indices.size(); i++) {
        std::string name = "indices" + std::to_string(i);
        cmd.Input(indices[i], name);
    }
    cmd.DynamicInputReg(index_func, {{indices.size(), 3}});
    cmd.Output(result);
    if (!is_aicore) {
        cmd.Attr("_exclude_engines", aicore_str);
    }
    cmd.Run();
    return result;
}

at::Tensor index_high_dims(const at::Tensor &self, std::vector<at::Tensor> indices)
{
    // masks corresponds to indices. 0 indicates undefined tensor.
    at::SmallVector<int64_t, N> masks;
    std::vector<at::Tensor> all_defined_indices;
    for (uint64_t i = 0; i < indices.size(); i++) {
        if (indices[i].defined()) {
            all_defined_indices.emplace_back(indices[i]);
            masks.emplace_back(1);
            continue;
        }
        masks.emplace_back(0);
    }

    /**
     * When input.size(0) = 1, if the dtype of indices is int64,
     * and indices only for 0 dimension, can broadcast to output.
     */
    if (self.size(0) == 1 && masks.size() == 1 && masks[0] == 1 && all_defined_indices[0].scalar_type() == at::kLong &&
        all_defined_indices[0].dim() == 1) {
        c10::SmallVector<int64_t, N> output_size = op_infer::array_to_small_vector(self.sizes());
        output_size[0] = all_defined_indices[0].size(0);
        at::Tensor result = acl_op::npu_broadcast(self, output_size);
        return result;
    }

    at::Tensor self_nd = at_npu::native::custom_ops::npu_format_cast(self, ACL_FORMAT_ND);

    bool is_aicore = check_index_aicore(all_defined_indices, masks);
    bool is_casted = false;
    at::Tensor self_data = self_nd;

    if (is_aicore && (self.scalar_type() == at::kByte || self.scalar_type() == at::kBool)) {
        is_casted = true;
        self_data = at_npu::native::custom_ops::npu_dtype_cast(self_nd, at::kInt);
    }
    auto output_size = op_infer::index_npu_output_size(self_data, indices);
    auto result = npu_preparation::apply_tensor_with_format(self_data, output_size, ACL_FORMAT_ND);

    index_out_nocheck(self_data, masks, all_defined_indices, result, is_aicore);

    if (is_casted) {
        auto result_casted = at_npu::native::custom_ops::npu_dtype_cast(result, self.scalar_type());
        return result_casted;
    }

    return result;
}
} // namespace

at::Tensor index_common(const at::Tensor &self, const torch::List<c10::optional<at::Tensor>> &orig)
{
    bool needCast = op_plugin::AdvanceIndex::checkIndexTensorTypes(orig);
    auto indices = op_plugin::AdvanceIndex::npu_expand_tensors(self, orig, needCast);
    auto broadcast_indices = op_plugin::AdvanceIndex::npu_broadcast_tensors(indices);

    // not to transpose at all scene
    return index_high_dims(self, broadcast_indices);
}
} // namespace acl_op
