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
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static at::Tensor self_tensor_to_device(const at::Tensor &tensor, const at::ScalarType result_type,
                                        const c10::Device device)
{
    if (npu_preparation::is_scalar_wrapped_to_tensor(tensor) ||
        (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor))) {
        at::Scalar scalar = tensor.item();
        return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
    }
    return tensor;
}

static at::Tensor pow_dest_output(const at::Tensor &self, const at::Tensor &exponent)
{
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return isSelfWrapped ? exponent : self;
}

static at::Tensor &pow_out_npu_nocheck(const at::Tensor &self, const at::Tensor &exponent, at::Tensor &out)
{
    if (exponent.dim() == 0 && !torch_npu::utils::is_npu(exponent)) {
        c10::Scalar exponent_scalar = exponent.item();
        EXEC_NPU_CMD(aclnnPowTensorScalar, self, exponent_scalar, out);
    } else {
        EXEC_NPU_CMD(aclnnPowTensorTensor, self, exponent, out);
    }
    return out;
}

static at::Tensor &inplace_pow_out_npu_nocheck(at::Tensor &self, const at::Tensor &exponent)
{
    if (exponent.dim() == 0 && !torch_npu::utils::is_npu(exponent)) {
        c10::Scalar exponent_scalar = exponent.item();
        EXEC_NPU_CMD(aclnnInplacePowTensorScalar, self, exponent_scalar);
    } else {
        EXEC_NPU_CMD(aclnnInplacePowTensorTensor, self, exponent);
    }
    return self;
}

at::Tensor pow(const at::Tensor &self, const at::Tensor &exponent)
{
    DO_COMPATIBILITY(aclnnPowTensorTensor, acl_op::pow(self, exponent));
    std::vector<at::Tensor> tensor_list = {self, exponent};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    // calculate the output size
    at::Tensor output_tensor = pow_dest_output(self, exponent);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, exponent);
    at::ScalarType result_type = at::native::result_type(self, exponent);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, output_tensor.device());
    // construct the output tensor of the NPU
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size,
                                                                  output_tensor.options().dtype(result_type));
    // calculate the output result of the NPU
    pow_out_npu_nocheck(self_cp, exponent, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor &pow_out(const at::Tensor &self, const at::Tensor &exponent, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnPowTensorTensor, acl_op::pow_out(self, exponent, out));
    std::vector<at::Tensor> tensor_list = {self, exponent};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    // calculate the output size
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, exponent);
    at::ScalarType result_type = out.scalar_type();
    at::Tensor self_cp = self_tensor_to_device(self, result_type, out.device());
    npu_preparation::check_tensor({self, exponent}, out, result_type, output_size);
    // calculate the output result of the NPU
    pow_out_npu_nocheck(self_cp, exponent, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor &pow_(at::Tensor &self, const at::Tensor &exponent)
{
    DO_COMPATIBILITY(aclnnInplacePowTensorTensor, acl_op::pow_(self, exponent));
    std::vector<at::Tensor> tensor_list = {self, exponent};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    inplace_pow_out_npu_nocheck(self, exponent);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}
}
