// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

at::Tensor& atan2_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnAtan2, acl_op::atan2_out(self, other, out));
    at::Tensor cp_self = self;
    if (npu_preparation::IsCPUScalar(self)) {
        at::Scalar scalar = self.item();
        cp_self = npu_preparation::copy_scalar_to_device(scalar, self.scalar_type(), other.device());
    }
    at::Tensor cp_other = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        cp_other = npu_preparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size = op_infer::broadcast_ops_npu_output_size(cp_self, cp_other);
    npu_preparation::check_tensor({cp_self, cp_other}, out, out.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnAtan2, cp_self, cp_other, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor atan2(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnAtan2, acl_op::atan2(self, other));
    at::Tensor cp_self = self;
    if (npu_preparation::IsCPUScalar(self)) {
        at::Scalar scalar = self.item();
        cp_self = npu_preparation::copy_scalar_to_device(scalar, self.scalar_type(), other.device());
    }
    at::Tensor cp_other = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        cp_other = npu_preparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size = op_infer::broadcast_ops_npu_output_size(cp_self, cp_other);
    c10::ScalarType infer_dtype = at::native::result_type(cp_self, cp_other);
    auto out_dtype = infer_dtype;
    if (isIntegralType(infer_dtype, true)) {
        out_dtype = at::kFloat;
    }
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, cp_self.options().dtype(out_dtype));
    EXEC_NPU_CMD(aclnnAtan2, cp_self, cp_other, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor& atan2_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceAtan2, acl_op::atan2_(self, other));
    at::Tensor cp_other = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        cp_other = npu_preparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnInplaceAtan2, self, cp_other);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}

}
