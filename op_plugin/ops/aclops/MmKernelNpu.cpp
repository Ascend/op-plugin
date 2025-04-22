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
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
const int DIMENSION_2D = 2;
const int DIMENSION_3D = 3;
using npu_preparation = at_npu::native::OpPreparation;
using format_helper = at_npu::native::FormatHelper;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
/*****************************************
Function: is_transpose_last_two_dims_flex
Description:
  Flexible transpose judgement for view+transpose+Matmul, i.e.,
  tensors with dim=2 and base_size_.size=n can also be Matmul directly!
Return:
  True--Cases are flex transposed(flex transpose=strict transpose+view
    transpose), which can be refreshed as a input transposed tensor proceed to
Matmul: [1] 2-2-t(strict transpose); [2] 2-n-view+t(view transpose).
  False--Tensor is not transposed, proceed to format_contiguous.
*****************************************/
bool is_transpose_last_two_dims_flex(const at::Tensor &tensor)
{
    if (tensor.dim() != DIMENSION_2D) {
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
    if (is_transpose_flex && base_sizes.size() == static_cast<uint>(tensor.dim()) &&
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

void mm_insert_input_transpose(at::Tensor &tensor, bool &is_tensor_trans_flex, bool &is_tensor_trans_strict)
{
    tensor = is_tensor_trans_flex ? tensor.clone() : tensor.transpose(-1, -2).clone();
    is_tensor_trans_flex = !is_tensor_trans_flex;
    is_tensor_trans_strict = !is_tensor_trans_strict;
}

void mm_set_format_contiguous(at::Tensor &tensor, const bool &is_tensor_trans_flex, const bool &is_tensor_trans_strict)
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

bool is_mm_transpose(const at::Tensor &tensor)
{
    if (tensor.dim() < DIMENSION_2D || tensor.dim() > DIMENSION_3D) {
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

bool mm_check_nd_to_nz_on_the_fly(const at::Tensor &self, const at::Tensor &mat2)
{
    const static int64_t kInnerAxisMinBytes = 256;
    const static int64_t kInnerAxisMaxLimit = 65535;
    int64_t self_inner_axis = self.size(self.dim() - 1);
    int64_t self_outer_axis = self.size(self.dim() - 2);
    int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
    int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
    if (is_mm_transpose(self)) {
        self_inner_axis = self.size(self.dim() - 2);
        self_outer_axis = self.size(self.dim() - 1);
    }
    if (is_mm_transpose(mat2)) {
        mat2_inner_axis = mat2.size(mat2.dim() - 2);
        mat2_outer_axis = mat2.size(mat2.dim() - 1);
    }
    int64_t data_type = static_cast<int64_t>(elementSize(self.scalar_type()));
    if (self_outer_axis > kInnerAxisMaxLimit && self_inner_axis * data_type < kInnerAxisMinBytes &&
        bool((static_cast<uint64_t>(self_inner_axis) * static_cast<uint64_t>(data_type)) & 0x1F)) {
        return false;
    }
    return !((self_inner_axis > kInnerAxisMaxLimit && self_outer_axis > kInnerAxisMaxLimit) ||
             (mat2_inner_axis > kInnerAxisMaxLimit && mat2_outer_axis > kInnerAxisMaxLimit));
}

bool is_transpose_inner_axis(const at::Tensor &self)
{
    const static int64_t kInnerAxisMinBytes = 256;
    const static int64_t kInnerAxisMaxLimit = 65535;
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1 || self.dim() < DIMENSION_2D ||
        (self.scalar_type() != at::ScalarType::Half && self.scalar_type() != at::ScalarType::Float &&
         self.scalar_type() != at::ScalarType::BFloat16)) {
        return false;
    }
    int64_t data_type = static_cast<int64_t>(elementSize(self.scalar_type()));
    int64_t self_inner_axis = self.size(self.dim() - 1);
    int64_t self_outer_axis = self.size(self.dim() - 2);
    if (is_mm_transpose(self)) {
        self_inner_axis = self.size(self.dim() - 2);
        self_outer_axis = self.size(self.dim() - 1);
    }
    if (self_inner_axis == 1 && self_outer_axis > kInnerAxisMaxLimit) {
        return true;
    }
    if (self_inner_axis * self_outer_axis <= kInnerAxisMaxLimit) {
        // too small tensor size
        return false;
    }
    return ((self_inner_axis > kInnerAxisMaxLimit) ||
            (self_inner_axis * data_type < kInnerAxisMinBytes &&
             bool((static_cast<uint64_t>(self_inner_axis) * static_cast<uint64_t>(data_type)) & 0x1F))) &&
           ((self_outer_axis * data_type >= kInnerAxisMinBytes && self_outer_axis <= kInnerAxisMaxLimit) ||
            (self_outer_axis * data_type < kInnerAxisMinBytes &&
             ((static_cast<uint64_t>(self_outer_axis) * static_cast<uint64_t>(data_type)) & 0x1F) == 0));
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

int64_t ceil(int64_t x, int64_t y)
{
    TORCH_CHECK(y != 0, "Error, zero division." + OPS_ERROR(ErrCode::VALUE));
    return ((x + y - 1) / y) * y;
}

int64_t ceil_div(int64_t x, int64_t y)
{
    TORCH_CHECK(y != 0, "Error, zero division." + OPS_ERROR(ErrCode::VALUE));
    return (x + y - 1) / y;
}


at::Tensor &mm_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &mat2)
{
    const auto self_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    const auto mat2_desc = torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_;
    bool is_self_t_flex = is_transpose_last_two_dims_flex(self);
    bool is_mat2_t_flex = is_transpose_last_two_dims_flex(mat2);
    bool is_self_t_strict = is_transpose_last_two_dims_strict(self, is_self_t_flex);
    bool is_mat2_t_strict = is_transpose_last_two_dims_strict(mat2, is_mat2_t_flex);
    at::Tensor contiguous_self = self;
    at::Tensor contiguous_mat2 = mat2;

    bool is_transpose_self = is_transpose_inner_axis(contiguous_self);
    bool is_transpose_mat2 = is_transpose_inner_axis(contiguous_mat2);
    if (is_transpose_self && is_transpose_mat2 && !is_transpose_both_inner_axis(contiguous_self, contiguous_mat2)) {
        is_transpose_self = !is_transpose_self;
        is_transpose_mat2 = !is_transpose_mat2;
    }

    if (is_transpose_self) {
        mm_insert_input_transpose(contiguous_self, is_self_t_flex, is_self_t_strict);
    }
    if (is_transpose_mat2) {
        mm_insert_input_transpose(contiguous_mat2, is_mat2_t_flex, is_mat2_t_strict);
    }

    mm_set_format_contiguous(contiguous_self, is_self_t_flex, is_self_t_strict);
    mm_set_format_contiguous(contiguous_mat2, is_mat2_t_flex, is_mat2_t_strict);

    at_npu::native::OpCommand cmd;

    try {
        cmd.Name("MatMul")
            .InputWithoutContiguous(contiguous_self)
            .InputWithoutContiguous(contiguous_mat2)
            .Output(result)
            .Attr("transpose_x1", is_self_t_flex)
            .Attr("transpose_x2", is_mat2_t_flex)
            .Run();
    } catch (...) {
        // Recover storage desc of view-transpose tensors, i.e. the inverse process of
        // set_transposed_npu_desc
        if (is_self_t_flex && (!is_self_t_strict)) {
            torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_ = self_desc;
        }
        if (is_mat2_t_flex && (!is_mat2_t_strict)) {
            torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
        }
        throw;
    }

    // Recover storage desc of view-transpose tensors, i.e. the inverse process of
    // set_transposed_npu_desc
    if (is_self_t_flex && (!is_self_t_strict)) {
        torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_ = self_desc;
    }
    if (is_mat2_t_flex && (!is_mat2_t_strict)) {
        torch_npu::NPUBridge::GetNpuStorageImpl(mat2)->npu_desc_ = mat2_desc;
    }

    return result;
}
} // namespace

at::Tensor &mm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    TORCH_CHECK(self.dim() == DIMENSION_2D && mat2.dim() == DIMENSION_2D,
                "both arguments to matmul need to be 2D, but they are ",
                self.dim(), "D and ", mat2.dim(), "D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Char && mat2.scalar_type() != at::ScalarType::Char,
                "mm_out is not support int8 dtype", OPS_ERROR(ErrCode::PARAM))
    if (!out.is_contiguous()) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out);
        mm_out_npu_nocheck(contiguous_out, self, mat2);
        npu_utils::format_fresh_view(out, contiguous_out);
    } else {
        mm_out_npu_nocheck(out, self, mat2);
    }
    return out;
}

at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2)
{
    TORCH_CHECK(self.dim() == DIMENSION_2D && mat2.dim() == DIMENSION_2D,
                "both arguments to matmul need to be 2D, but they are ",
                self.dim(), "D and ", mat2.dim(), "D", OPS_ERROR(ErrCode::PARAM));

    // 1.cann bmm support int8(input)->int32(out)
    // 2.onnx can support because of change y dtype to be int32.
    // 3.torch need int8(input)->int8(out), cann can not support.
    TORCH_CHECK(self.scalar_type() != at::ScalarType::Char && mat2.scalar_type() != at::ScalarType::Char,
                "mm is not support int8 dtype", OPS_ERROR(ErrCode::PARAM))
    auto output_size = {self.size(0), mat2.size(1)};

    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
    bool need_nd_out = false;
    static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
    // check format_out of mm is NCHW. Delate after definite NLP model.
    if ((self.scalar_type() == at::ScalarType::Half)) {
        // check is 16-algined with high-performance
        auto is_align = [&]() {
            return ((static_cast<uint64_t>(self.size(0)) & 0xF) == 0) &&
                   ((static_cast<uint64_t>(self.size(1)) & 0xF) == 0) &&
                   ((static_cast<uint64_t>(mat2.size(0)) & 0xF) == 0) &&
                   ((static_cast<uint64_t>(mat2.size(1)) & 0xF) == 0);
        };
        // There is a data trampling problem in non-aligned scenes. For the time
        // being, only aligned scenes are supported.
        static auto mm_bmm_nd = !at_npu::native::env::CheckMmBmmNDDisable();
        if (format_helper::IsBaseFormatType(self) && format_helper::IsBaseFormatType(mat2) && mm_bmm_nd &&
            ((is_support_nd_out && mm_check_nd_to_nz_on_the_fly(self, mat2)) || (!is_support_nd_out && is_align()))) {
            result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
        } else {
            need_nd_out = mm_bmm_nd;
            result =
                npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_FRACTAL_NZ, true);
        }
    }

    mm_out_npu_nocheck(result, self, mat2);

    if (need_nd_out) {
        result = at_npu::native::custom_ops::npu_format_cast(result, ACL_FORMAT_ND);
    }
    return result;
}
} // namespace acl_op
