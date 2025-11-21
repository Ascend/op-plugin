// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

std::tuple<at::Tensor, at::Tensor> batch_norm_reduce(const at::Tensor& self, double eps)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) {
        return acl_op::batch_norm_reduce(self, eps);
    } else {
        TORCH_CHECK(self.dim() > 1, "The dim input tensor [self] must more than 1." + OPS_ERROR(ErrCode::PARAM));

        int64_t c_value;
        c10::SmallVector<int64_t, N> dimlist;
        int64_t self_npu_format = npu_preparation::get_tensor_npu_format(self);
        if (self_npu_format == ACL_FORMAT_NHWC || self_npu_format == ACL_FORMAT_NDHWC) {
            c_value = self.size(-1);
            dimlist = c10::SmallVector<int64_t, N>{-1};
        } else {
            c_value = self.size(1);
            c10::SmallVector<int64_t, N> all_dim = op_plugin::utils::get_dimlist_for_tensor(self);
            auto it = std::remove(all_dim.begin(), all_dim.end(), 1);
            all_dim.erase(it, all_dim.end());
            dimlist = all_dim;
        }

        auto output_size = {c_value};
        at::Tensor mul_out = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(at::kFloat), self);
        at::Tensor sum = npu_preparation::apply_tensor(output_size, self.options().dtype(at::kFloat), self);
        at::Tensor square_sum = npu_preparation::apply_tensor(output_size, self.options().dtype(at::kFloat), self);

        at::Tensor self_copy = self;
        if (self.scalar_type() != at::kFloat) {
            self_copy = at_npu::native::custom_ops::_npu_dtype_cast(self_copy, at::kFloat);
        }

        EXEC_NPU_CMD(aclnnMul, self_copy, self_copy, mul_out);
        sum = op_api::sum(self_copy, dimlist, false, at::kFloat);
        square_sum = op_api::sum(mul_out, dimlist, false, at::kFloat);

        return std::make_tuple(sum, square_sum);
    }
}
}