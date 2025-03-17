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

#ifndef PULGIN_UTILS_ADVANCED_INDEX
#define PULGIN_UTILS_ADVANCED_INDEX

#include <ATen/native/IndexingUtils.h>
#include <ATen/ExpandUtils.h>

#include "op_plugin/utils/Export.h"

namespace op_plugin {

struct OP_PLUGIN_HIDDEN AdvancedIndex {
    AdvancedIndex(const at::Tensor& src, at::TensorList list_indices);
    at::Tensor src;
    std::vector<at::Tensor> indices;
    at::DimVector indexed_sizes;
    at::DimVector indexed_strides;
    int64_t dims_before;
    int64_t dims_after;
};

class OP_PLUGIN_HIDDEN AdvanceIndex {
public:
    static bool all_strides_match(at::TensorList tensor_list);
    static at::Tensor reshape_indexer(const at::Tensor& index, int64_t dims_before, int64_t dims_after);
    static at::Tensor restride_src(const at::Tensor& src, int64_t before_dims, int64_t dims_indexed,
        at::IntArrayRef replacement_shape);
    static std::string shapes_as_str(at::TensorList tensors);
    static AdvancedIndex make_info(at::Tensor self, const torch::List<c10::optional<at::Tensor>>& orig);
    static std::vector<at::Tensor> npu_expand_tensors(
        const at::Tensor& self,
        const torch::List<c10::optional<at::Tensor>>& indices,
        bool needCast,
        bool flag_aclnn = false);
    static std::vector<at::Tensor> npu_broadcast_tensors(std::vector<at::Tensor> to_broadcast);
    static bool is_expandable_to(c10::IntArrayRef shape, c10::IntArrayRef desired);
    static bool checkIndexTensorTypes(const torch::List<c10::optional<at::Tensor>> &indices);
};

} // namespace op_plugin
#endif
