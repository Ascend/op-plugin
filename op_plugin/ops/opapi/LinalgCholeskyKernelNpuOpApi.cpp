// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

const int DIM_2D = 2;

at::Tensor& linalg_cholesky_out(const at::Tensor& self, bool upper, at::Tensor& out)
{
    if (!check_aclnn_kernel_available("aclnnLinalgCholesky")) {
        auto info = at::empty({0}, self.options().dtype(at::kInt));
        at::linalg_cholesky_ex_out(out, info, self, upper, false);
        at::_linalg_check_errors(info, "linalg.cholesky", self.dim() == DIM_2D);
        return out;
    }
    auto output_size = self.sizes();
    auto output_dtype = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype, output_size);
    EXEC_NPU_CMD(aclnnLinalgCholesky, self, upper, out);
    return out;
}

at::Tensor linalg_cholesky(const at::Tensor& self, bool upper)
{
    if (!check_aclnn_kernel_available("aclnnLinalgCholesky")) {
        auto [out, info] = at::linalg_cholesky_ex(self, upper, false);
        at::_linalg_check_errors(info, "linalg.cholesky", self.dim() == DIM_2D);
        return out;
    }
    auto output_size = self.sizes();
    auto output_dtype = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_dtype));
    EXEC_NPU_CMD(aclnnLinalgCholesky, self, upper, out);
    return out;
}
}