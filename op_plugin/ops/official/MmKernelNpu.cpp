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

#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
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
bool is_transpose_last_two_dims_flex(const at::Tensor& tensor) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
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
bool is_transpose_last_two_dims_strict(const at::Tensor& tensor, bool is_transpose_flex) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    return false;
  }
  auto base_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
  if (is_transpose_flex && base_sizes.size() == tensor.dim() &&
      tensor.size(-1) == base_sizes[tensor.dim() - 2] &&
      tensor.size(-2) == base_sizes[tensor.dim() - 1]) {
    return true;
  }
  return false;
}

// Refresh storage desc of view-transpose tensor.
void set_transposed_npu_desc(at::Tensor& tensor) {
  at::Tensor temp_transpose_Tensor = tensor.transpose(-1, -2);
  at_npu::native::StorageDescHelper::SetDesc(tensor, temp_transpose_Tensor.sizes(), temp_transpose_Tensor.strides());
}

at::Tensor& mm_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat2) {
  const auto& self_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(self);
  const auto& mat2_desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(mat2);
  bool is_self_t_flex = is_transpose_last_two_dims_flex(self);
  bool is_mat2_t_flex = is_transpose_last_two_dims_flex(mat2);
  bool is_self_t_strict = is_transpose_last_two_dims_strict(self, is_self_t_flex);
  bool is_mat2_t_strict = is_transpose_last_two_dims_strict(mat2, is_mat2_t_flex);
  at::Tensor contiguous_self = self;
  at::Tensor contiguous_mat2 = mat2;

  if (is_self_t_flex) {
    if (!is_self_t_strict) {
      // Matmul cannot directly deal with view+transposed tensor with NZ format,
      // so Transdata is necessary
      contiguous_self = npu_preparation::CastBackToOriFormat(self);
      // Storage desc of view-transpose tensors should be refreshed to be
      // matched.
      set_transposed_npu_desc(contiguous_self);
    }
  } else {
    contiguous_self = npu_utils::format_contiguous_add_copy_optimize(self);
  }

  if (is_mat2_t_flex) {
    if (!is_mat2_t_strict) {
      // Matmul cannot directly deal with view+transposed tensor with NZ format,
      // so Transdata is necessary
      contiguous_mat2 = npu_preparation::CastBackToOriFormat(mat2);
      // Storage desc of view-transpose tensors should be refreshed to be
      // matched.
      set_transposed_npu_desc(contiguous_mat2);
    }
  } else {
    contiguous_mat2 = npu_utils::format_contiguous_add_copy_optimize(mat2);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("MatMul")
      .InputWithoutContiguous(contiguous_self)
      .InputWithoutContiguous(contiguous_mat2)
      .Output(result)
      .Attr("transpose_x1", is_self_t_flex)
      .Attr("transpose_x2", is_mat2_t_flex)
      .Run();

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

at::Tensor& mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    mm_out_npu_nocheck(contiguous_result, self, mat2);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    mm_out_npu_nocheck(result, self, mat2);
  }
  return result;
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  const static int SPLIT_K_MULTI = 8;
  auto output_size = {self.size(0), mat2.size(1)};
  auto k_dim = self.size(1);

  at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
  bool need_nd_out = false;
  static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
  bool split_k = is_support_nd_out && 
      (k_dim >= SPLIT_K_MULTI * std::max(self.size(0), mat2.size(1))) &&
      (self.dtype() == at::ScalarType::Half) && (mat2.dtype() == at::ScalarType::Half) &&
      (format_helper::GetFormat(self) == ACL_FORMAT_ND) &&
      (format_helper::GetFormat(mat2) == ACL_FORMAT_ND);
  // check format_out of mm is NCHW. Delate after definite NLP model.
  if ((self.scalar_type() == at::ScalarType::Half)) {
    // check is 16-algined with high-performance
    auto is_aligin = [&]() {
      return (!(static_cast<uint64_t>(self.size(0)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(self.size(1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(0)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2.size(1)) & 0x0000000F));
    };
    // There is a data trampling problem in non-aligned scenes. For the time
    // being, only aligned scenes are supported.
    static auto mm_bmm_nd = !at_npu::native::env::CheckMmBmmNDDisable();
    if (format_helper::IsBaseFormatType(self) && format_helper::IsBaseFormatType(mat2)
        && mm_bmm_nd && ((is_support_nd_out && calcu_op_util::IsNdToNzOnTheFly(self, mat2)) ||
        (!is_support_nd_out && is_aligin()))) {
      if (split_k) {
        result = npu_preparation::apply_tensor_with_format(
            output_size,
            self.options().dtype(at::ScalarType::Float),
            ACL_FORMAT_ND);
      } else {
        result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_ND);
      }
    } else {
      need_nd_out = mm_bmm_nd;
      if (split_k) {
        result = npu_preparation::apply_tensor_with_format(
            output_size,
            self.options().dtype(at::ScalarType::Float),
            ACL_FORMAT_FRACTAL_NZ,
            true);
      } else {
        result = npu_preparation::apply_tensor_with_format(output_size, self.options(), ACL_FORMAT_FRACTAL_NZ, true);
      }
    }
  }

  mm_out_npu_nocheck(result, self, mat2);

  if (need_nd_out) {
    result = at_npu::native::NPUNativeFunctions::npu_format_cast(result, ACL_FORMAT_ND);
  }
  result = split_k ? op_plugin::npu_dtype_cast(result, at::ScalarType::Half) : result;
  return result;
}
} // namespace op_plugin
