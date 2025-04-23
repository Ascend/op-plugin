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
#include "op_plugin/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
using format_helper = at_npu::native::FormatHelper;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

bool is_transpose_last_two_dims_flex(const at::Tensor &tensor)
{
    if (tensor.dim() != 2) {
        return false;
    }

    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        return true;
    } else {
        return false;
    }
}

// Pick out strict-transpose tensors from flex-transpose tensors.
bool is_transpose_last_two_dims_strict(const at::Tensor &tensor, bool is_transpose_flex)
{
    auto base_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
    if (is_transpose_flex && base_sizes.size() == static_cast<uint64_t>(tensor.dim()) &&
        tensor.size(-1) == base_sizes[tensor.dim() - 2] && tensor.size(-2) == base_sizes[tensor.dim() - 1]) {
        return true;
    }
    return false;
}

// Refresh storage desc of view-transpose tensor.
void set_transposed_npu_desc(at::Tensor &tensor)
{
    at::Tensor temp_transpose_Tensor = tensor.transpose(-1, -2);
    at_npu::native::StorageDescHelper::SetDesc(tensor, temp_transpose_Tensor.sizes(), temp_transpose_Tensor.strides());
}

void mm_set_format_contiguous(at::Tensor &tensor, bool &is_tensor_trans_flex, bool &is_tensor_trans_strict)
{
    if (is_tensor_trans_flex) {
        if (!is_tensor_trans_strict) {
            // Matmul cannot directly deal with view+transposed tensor with NZ format,
            // so Transdata is necessary
            tensor = npu_preparation::CastBackToOriFormat(tensor);
            // Storage desc of view-transpose tensors should be refreshed to be
            // matched.
            set_transposed_npu_desc(tensor);
        }
    } else {
        tensor = npu_utils::format_contiguous_add_copy_optimize(tensor);
    }
}

bool is_transpose_both_inner_axis(const at::Tensor &self, const at::Tensor &mat2)
{
    const static int64_t kInnerAxisMaxLimit = 65535;
    int64_t self_inner_axis = self.size(self.dim() - 1);
    int64_t self_outer_axis = self.size(self.dim() - 2);
    int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
    int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
    if (op_plugin::utils::is_transpose_last_two_dims(self)) {
        self_inner_axis = self.size(self.dim() - 2);
        self_outer_axis = self.size(self.dim() - 1);
    }
    if (op_plugin::utils::is_transpose_last_two_dims(mat2)) {
        mat2_inner_axis = mat2.size(mat2.dim() - 2);
        mat2_outer_axis = mat2.size(mat2.dim() - 1);
    }
    return self_inner_axis > kInnerAxisMaxLimit && self_outer_axis <= kInnerAxisMaxLimit &&
           mat2_inner_axis > kInnerAxisMaxLimit && mat2_outer_axis <= kInnerAxisMaxLimit;
}

at::Tensor &addmm_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &mat1,
                                  const at::Tensor &mat2)
{
    const auto mat1_desc = torch_npu::NPUBridge::GetNpuStorageImpl(mat1)->npu_desc_;
    const auto mat2_desc = torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_;
    bool is_mat1_t_flex = is_transpose_last_two_dims_flex(mat1);
    bool is_mat2_t_flex = is_transpose_last_two_dims_flex(mat2);
    bool is_mat1_t_strict = is_transpose_last_two_dims_strict(mat1, is_mat1_t_flex);
    bool is_mat2_t_strict = is_transpose_last_two_dims_strict(mat2, is_mat2_t_flex);
    at::Tensor contiguous_mat1 = mat1;
    at::Tensor contiguous_mat2 = mat2;
    mm_set_format_contiguous(contiguous_mat1, is_mat1_t_flex, is_mat1_t_strict);
    mm_set_format_contiguous(contiguous_mat2, is_mat2_t_flex, is_mat2_t_strict);
    at_npu::native::OpCommand cmd;
    try {
        cmd.Name("MatMul")
            .InputWithoutContiguous(contiguous_mat1)
            .InputWithoutContiguous(contiguous_mat2)
            .Input(self)
            .Output(result)
            .Attr("transpose_x1", is_mat1_t_flex)
            .Attr("transpose_x2", is_mat2_t_flex)
            .Run();
    } catch (...) {
        // Recover storage desc of view-transpose tensors, i.e. the inverse process of
        // set_transposed_npu_desc
        if (is_mat1_t_flex && (!is_mat1_t_strict)) {
            torch_npu::NPUBridge::GetNpuStorageImpl(mat1)->npu_desc_ = mat1_desc;
        }
        if (is_mat2_t_flex && (!is_mat2_t_strict)) {
            torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
        }
        throw;
    }

    // Recover storage desc of view-transpose tensors, i.e. the inverse process of
    // set_transposed_npu_desc
    if (is_mat1_t_flex && (!is_mat1_t_strict)) {
        torch_npu::NPUBridge::GetNpuStorageImpl(mat1)->npu_desc_ = mat1_desc;
    }
    if (is_mat2_t_flex && (!is_mat2_t_strict)) {
        torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
    }
    return result;
}

at::Tensor &addmm_out(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                      const at::Scalar &alpha, at::Tensor &result)
{
    static const bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
    bool check_bias_shape = (self.dim() == 1 || (self.dim() == 2 && self.size(0) == 1));
    if (check_bias_shape && is_support_nd_out) {
        if (beta.toFloat() == 1.0 && alpha.toFloat() == 1.0) {
            acl_op::addmm_out_npu_nocheck(result, self, mat1, mat2);
        } else {
            at::Tensor mul_result = at::mul(mat1, alpha);
            at::Tensor bias = at::mul(self, beta);
            acl_op::addmm_out_npu_nocheck(result, bias, mul_result, mat2);
        }
    } else {
        at::Tensor mul_result = at::mul(mat1, alpha);
        at::Tensor mm_result = at::mm(mul_result, mat2);
        // matmul*alpha+self*beta
        at::add_out(result, mm_result, self, beta);
    }
    return result;
}

at::Tensor addmm(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                 const at::Scalar &alpha)
{
    auto output_size = op_infer::addmm_npu_output_size(self, mat1, mat2);
    at::Tensor result = npu_preparation::apply_tensor(output_size, self.options(), self);

    acl_op::addmm_out(self, mat1, mat2, beta, alpha, result);
    return result;
}

at::Tensor &addmm_(at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2, const at::Scalar &beta,
                   const at::Scalar &alpha)
{
    npu_preparation::CheckMemory({self, mat1, mat2}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        acl_op::addmm_out(contiguous_self, mat1, mat2, beta, alpha, contiguous_self);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        acl_op::addmm_out(self, mat1, mat2, beta, alpha, self);
    }
    return self;
}
} // namespace acl_op
