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
using npu_preparation = at_npu::native::OpPreparation;

static const int64_t MIN_DEPTH = 1;
static const int64_t AUTO_DEPTH = -1;
static const int64_t MIN_NUM_CLASSES = 0;

at::Tensor one_hot(const at::Tensor& self, int64_t num_classes)
{
    DO_COMPATIBILITY(aclnnOneHot, acl_op::one_hot(self, num_classes));
    int64_t depth = num_classes;
    TORCH_CHECK(depth >= AUTO_DEPTH, "NPU error, not yet support negative num_classes, when num_classes less than -1",
                OPS_ERROR(ErrCode::PARAM));
    // when the self is empty, num_classes should be greater than 0
    TORCH_CHECK(self.numel() != 0 || num_classes > MIN_NUM_CLASSES,
                "NPU error, can not infer total number of classes from empty tensor.", OPS_ERROR(ErrCode::PARAM));
    if (depth == AUTO_DEPTH) {
        depth = self.max().item().toLong() + 1;
        if (depth < MIN_DEPTH) {
            depth = MIN_DEPTH;
        }
    }
    // construct on_value tensor
    at::Tensor on_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    on_value_tensor.fill_(1);
    // construct off_value tensor
    at::Tensor off_value_tensor = npu_preparation::apply_tensor_without_format({1}, self.options());
    off_value_tensor.fill_(0);
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size.emplace_back(depth);
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor(output_size, self.options(), self);
    int64_t axis = -1;
    EXEC_NPU_CMD(aclnnOneHot, self, depth, on_value_tensor, off_value_tensor, axis, result);
    return result;
}
}
