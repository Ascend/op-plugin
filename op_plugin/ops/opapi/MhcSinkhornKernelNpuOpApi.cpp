// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using npu_utils = at_npu::native::NpuUtils;

tensor_list npu_mhc_sinkhorn_symint(const at::Tensor &x, double eps, c10::SymInt num_iters, int64_t out_flag)
{
    float eps_f = static_cast<float>(eps);
    int64_t num_iters_int =  num_iters.expect_int();
    at::Tensor result = npu_preparation::apply_tensor_with_format(x.sizes(), x.options(), ACL_FORMAT_ND);
    at::Tensor norm_out;
    at::Tensor sum_out;
    if (out_flag == 1) {
        int64_t T = x.size(0);
        if (x.dim() == 4) {
            T = T * x.size(1);
        }
        int64_t n = x.size(-1);
        c10::SmallVector<int64_t, SIZE> norm_out_size = {2 * num_iters_int * T * n * 8};
        c10::SmallVector<int64_t, SIZE> sum_out_size = {2 * num_iters_int * T * 8};
        norm_out = npu_preparation::apply_tensor_with_format(norm_out_size, x.options(), ACL_FORMAT_ND);
        sum_out = npu_preparation::apply_tensor_with_format(sum_out_size, x.options(), ACL_FORMAT_ND);
    }

    EXEC_NPU_CMD(aclnnMhcSinkhorn, x, eps_f, num_iters_int, result, norm_out, sum_out);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(result, norm_out, sum_out);
}

}  // namespace op_api