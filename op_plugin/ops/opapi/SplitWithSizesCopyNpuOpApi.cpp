// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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


std::vector<at::Tensor> split_with_sizes_copy(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim)
{
    auto output_shapes = op_infer::split_with_sizes_copy_output_size(op_infer::array_to_small_vector(self.sizes()), split_sizes, dim);
    auto output_dtype = self.scalar_type();

    std::vector<at::Tensor> result;
    for (const auto& shape : output_shapes) {
        result.push_back(npu_preparation::apply_tensor_without_format(shape, self.options().dtype(output_dtype)));
    }

    at::TensorList result_ = at::TensorList(result);
    EXEC_NPU_CMD(aclnnSplitWithSize, self, split_sizes, dim, result_);
    return result;
}

}  // namespace op_api
