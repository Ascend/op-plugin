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

at::Tensor _pdist_forward(const at::Tensor& self, double p)
{
    DO_COMPATIBILITY(aclnnPdist, acl_op::_pdist_forward(self, p));
    TORCH_CHECK(p >= 0, "pdist only supports non-negative p values", OPS_ERROR(ErrCode::VALUE));
    // double is not supported in NPU,  type of P needs to be converted from double to float.
    float p_float;
    if (std::isinf(p)) {
        p_float = std::numeric_limits<float>::infinity();
    } else {
        TORCH_CHECK(p <= std::numeric_limits<float>::max(), "p dose not support float64 currently.",
            OPS_ERROR(ErrCode::TYPE));
        p_float = static_cast<float>(p);
    }
    auto output_size = op_infer::pdist_npu_output_size(self);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnPdist, self, p_float, result);
    return result;
}

at::Tensor pdist(const at::Tensor& self, double p)
{
    return at::_pdist_forward(self, p);
}

}
