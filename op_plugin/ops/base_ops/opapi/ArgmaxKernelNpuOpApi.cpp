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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static at::Tensor& argmax_exec(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim, at::Tensor& result,
                               bool out_mode)
{
    at::Tensor input = self.reshape({-1});
    int64_t realDim = 0;
    bool realKeepDim = false;
    if (dim.has_value()) {
        input = self;
        realDim = dim.value();
        realKeepDim = keepdim;
    }
    auto output_size = op_infer::reduce_ops_npu_output_size(input, realDim, realKeepDim);
    if (out_mode) {
        npu_preparation::check_tensor({self}, result, result, output_size);
    } else {
        result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));
    }
    
    EXEC_NPU_CMD(aclnnArgMax, input, realDim, realKeepDim, result);
    return result;
}


at::Tensor argmax(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnArgMax, acl_op::argmax(self, dim, keepdim));
    if (self.numel() == 0) {
        return self;
    }
    at::Tensor result;
    return argmax_exec(self, dim, keepdim, result, false);
}


at::Tensor& argmax_out(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnArgMax, acl_op::argmax_out(self, dim, keepdim, result));
    if (self.numel() == 0) {
        result = self;
        return result;
    }
    return argmax_exec(self, dim, keepdim, result, true);
}
} // namespace op_api
