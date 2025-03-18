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

static at::Tensor& argmin_exec(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim, at::Tensor& result,
                               bool out_mode)
{
    TORCH_CHECK(!(self.numel() == 0 && !(dim.has_value())), "Expected reduction dim to be specified for input.numl()==0"
                + OPS_ERROR(ErrCode::PARAM))
        at::Tensor input;
    int64_t real_dim = 0;
    bool real_keep_dim = false;
    if (dim.has_value()) {
        input = self;
        real_dim = dim.value();
        real_keep_dim = keepdim;
    } else {
        input = self.reshape({-1});
    }

    // calculate the output size
    auto output_size = op_infer::reduce_ops_npu_output_size(input, real_dim, real_keep_dim);

    if (out_mode) {
        npu_preparation::check_tensor({self}, result, result, output_size);
    } else {
        // construct the output tensor of the NPU
        result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));
    }

    EXEC_NPU_CMD(aclnnArgMin, input, real_dim, real_keep_dim, result);
    return result;
}

at::Tensor argmin(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnArgMin, acl_op::argmin(self, dim, keepdim));

    at::Tensor result;
    return argmin_exec(self, dim, keepdim, result, false);
}

at::Tensor& argmin_out(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnArgMin, acl_op::argmin_out(self, dim, keepdim, out));

    return argmin_exec(self, dim, keepdim, out, true);
}
}
