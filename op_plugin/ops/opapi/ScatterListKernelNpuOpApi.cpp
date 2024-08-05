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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    std::vector<at::Tensor> npu_scatter_list(
        at::TensorList self,
        const at::Tensor &indice,
        const at::Tensor &updates,
        const c10::optional<at::Tensor> &mask,
        c10::string_view reduce,
        int64_t axis)
    {
        std::string reduce_str = std::string(reduce);
        char *reduce_ptr = const_cast<char *>(reduce_str.c_str());
        // The attribute 'reduce' of ScatterList only supports setting it to 'update'.
        std::vector<at::Tensor> result;
        for (const at::Tensor &tensor : self) {
            result.push_back(tensor.clone());
        }
        at::TensorList result_ = at::TensorList(result);

        EXEC_NPU_CMD(aclnnScatterList, result_, indice, updates, mask, reduce_ptr, axis);

        return result;
    }

    void npu_scatter_list_(
        at::TensorList self,
        const at::Tensor &indice,
        const at::Tensor &updates,
        const c10::optional<at::Tensor> &mask,
        c10::string_view reduce,
        int64_t axis)
    {
        std::string reduce_str = std::string(reduce);
        char *reduce_ptr = const_cast<char *>(reduce_str.c_str());
        EXEC_NPU_CMD(aclnnScatterList, self, indice, updates, mask, reduce_ptr, axis);
        return;
    }

}
