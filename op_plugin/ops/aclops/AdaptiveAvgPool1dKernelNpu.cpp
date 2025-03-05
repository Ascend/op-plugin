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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
namespace {
void adaptive_avg_pool1d_check(
    const char* function_name,
    const char* argument_name,
    at::IntArrayRef x)
{
    TORCH_CHECK(
        x.size() == 1,
        function_name, "() argument '", argument_name,
        "' should contain one int (got ", x.size(), ")" + OPS_ERROR(ErrCode::PARAM));
}
} // namespace

at::Tensor adaptive_avg_pool1d(const at::Tensor& self, at::IntArrayRef output_size)
{
    at::checkDimRange("adaptive_avg_pool1d", at::TensorArg(self, "self", 1), 2, 4 /* exclusive */);
    adaptive_avg_pool1d_check("adaptive_avg_pool1d", "output_size", output_size);
    auto output = acl_op::adaptive_avg_pool2d(self.unsqueeze(-2), {1, output_size[0]});
    return output.squeeze(-2);
}

} // namespace acl_op
