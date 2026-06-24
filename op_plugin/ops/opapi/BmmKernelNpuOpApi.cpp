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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static bool is_normal_broadcast_expanded(const at::Tensor &t)
{
    return t.stride(0) == 0 && t.size(0) != 0;
}

static at::Tensor restore_broadcast_tensor(const at::Tensor &t)
{
    // Restore a broadcast-expanded [B, M, K] (stride(0)==0) back to [1, M, K].
    // stride(0) is set to size(1)*size(2) so that CANN recognizes standard
    // transpose patterns (e.g. strides [M*K, 1, K]) and handles them natively
    // via a flag instead of inserting an extra Transpose op.
    auto sizes = std::array<int64_t, 3>{1, t.size(1), t.size(2)};
    auto strides = std::array<int64_t, 3>{t.size(1) * t.size(2), t.stride(1), t.stride(2)};
    return t.as_strided(sizes, strides);
}

static bool is_compatible_impl_enabled()
{
    static auto compatible_env = std::getenv("TORCH_NPU_USE_COMPATIBLE_IMPL");
    return compatible_env != nullptr && std::string(compatible_env) == "1";
}

// A dim-1 slice (e.g. t[:, start:end, :]) of a 3D tensor leaves stride(0) untouched
// while shape(1) shrinks, breaking the contiguity between dim 0 and dim 1:
//   stride(0) == orig_M * K  !=  M' * K  == stride(1) * shape(1)
// Stride-based slices (t[:, ::step, :]) scale stride(1) so the product still matches.
static bool is_dim1_slice_non_contiguous(const at::Tensor &t)
{
    if (t.dim() != 3) return false;
    if (t.stride(0) == 0) return false;  // broadcast-expanded
    if (t.stride(2) != 1) return false;
    if (t.stride(1) != t.size(2)) return false;
    if (t.stride(0) == t.stride(1) * t.size(1)) return false;
    // CANN aclnnMatMul optimization requires shape(1) to be a divisor of 16.
    if (t.size(1) <= 1 || 16 % t.size(1) != 0) return false;
    return true;
}

// TODO: Remove this preprocess when CANN aclnnBatchMatMul supports non-contiguous input.
// In compatible mode, _matmul_impl expands tensors before bmm for batch broadcasting,
// making them non-contiguous (stride(0)==0). Since aclnnBatchMatMul doesn't support
// non-contiguous input, CANN inserts an extra BroadcastTo op, hurting performance.
// Here we restore expanded tensors to their pre-broadcast contiguous form
// (e.g., [64,m,n] -> [1,m,n]) to avoid the redundant op.
// Skip restore when both inputs are broadcast-expanded, as restoring both to batch=1
// would make aclnnBatchMatMul produce a batch=1 output that mismatches expected batch=b.
// When a broadcast-expanded tensor also has a non-contiguous matrix layout (e.g. transposed),
// aclnnBatchMatMul would additionally insert an extra Transpose op; restoring to batch=1
// lets it handle the transpose natively via a flag.
static std::pair<at::Tensor, at::Tensor> maybe_restore_broadcast(
    const at::Tensor &self, const at::Tensor &mat2)
{
    if (!is_compatible_impl_enabled()) {
        return {self, mat2};
    }
    bool both_broadcast = is_normal_broadcast_expanded(self) && is_normal_broadcast_expanded(mat2);
    if (both_broadcast) {
        return {self, mat2};
    }
    at::Tensor self_in = is_normal_broadcast_expanded(self) ? restore_broadcast_tensor(self) : self;
    at::Tensor mat2_in = is_normal_broadcast_expanded(mat2) ? restore_broadcast_tensor(mat2) : mat2;
    return {self_in, mat2_in};
}

at::Tensor &bmm_out(
    const at::Tensor &self,
    const at::Tensor &mat2,
    const at::ScalarType output_dtype,
    at::Tensor &result)
{
    TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3D tensor");

    DO_MATMUL_COMPATIBILITY(aclnnBatchMatMulWeightNz, aclnnBatchMatMul, self, mat2, acl_op::bmm_out(self, mat2, result));
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};
    npu_preparation::check_tensor({self, mat2}, result, output_dtype, output_size);

    // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should
    // use and functions such as hfloat32 can be enabled through this switch
    int8_t cube_math_type = op_plugin::utils::get_cube_math_type_with_passthrough();

    if (op_plugin::utils::is_nz_format(mat2) && !op_plugin::utils::is_nz_format(self)) {
        EXEC_NPU_CMD(aclnnBatchMatMulWeightNz, self, mat2, result, cube_math_type);
    } else if (is_compatible_impl_enabled() && is_dim1_slice_non_contiguous(self) &&
               is_normal_broadcast_expanded(mat2)) {
        // aclnnMatmul internally optimizes away the dim-1 slice on self,
        // avoiding the extra copy that aclnnBatchMatMul would require.
        auto mat2_input = mat2[0];
        EXEC_NPU_CMD(aclnnMatmul, self, mat2_input, result, cube_math_type);
    } else {
        auto [self_input, mat2_input] = maybe_restore_broadcast(self, mat2);
        EXEC_NPU_CMD(aclnnBatchMatMul, self_input, mat2_input, result, cube_math_type);
    }

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    return result;
}

at::Tensor &bmm_out(
    const at::Tensor &self,
    const at::Tensor &mat2,
    at::Tensor &result)
{
    TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3D tensor");

    DO_MATMUL_COMPATIBILITY(aclnnBatchMatMulWeightNz, aclnnBatchMatMul, self, mat2, acl_op::bmm_out(self, mat2, result));
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};
    npu_preparation::check_tensor({self, mat2}, result, self.scalar_type(), output_size);

    // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should
    // use and functions such as hfloat32 can be enabled through this switch
    int8_t cube_math_type = op_plugin::utils::get_cube_math_type_with_passthrough();

    if (op_plugin::utils::is_nz_format(mat2) && !op_plugin::utils::is_nz_format(self)) {
        EXEC_NPU_CMD(aclnnBatchMatMulWeightNz, self, mat2, result, cube_math_type);
    } else if (is_compatible_impl_enabled() && is_dim1_slice_non_contiguous(self) &&
               is_normal_broadcast_expanded(mat2)) {
        // aclnnMatmul internally optimizes away the dim-1 slice on self,
        // avoiding the extra copy that aclnnBatchMatMul would require.
        auto mat2_input = mat2[0];
        EXEC_NPU_CMD(aclnnMatmul, self, mat2_input, result, cube_math_type);
    } else {
        auto [self_input, mat2_input] = maybe_restore_broadcast(self, mat2);
        EXEC_NPU_CMD(aclnnBatchMatMul, self_input, mat2_input, result, cube_math_type);
    }

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    return result;
}

at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2, const at::ScalarType output_dtype)
{
    TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3D tensor");

    DO_MATMUL_COMPATIBILITY(aclnnBatchMatMulWeightNz, aclnnBatchMatMul, self, mat2, acl_op::bmm(self, mat2));

    // calculate the output size
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(output_dtype));

    int8_t cube_math_type = op_plugin::utils::get_cube_math_type_with_passthrough();
    if (op_plugin::utils::is_nz_format(mat2) && !op_plugin::utils::is_nz_format(self)) {
        EXEC_NPU_CMD(aclnnBatchMatMulWeightNz, self, mat2, result, cube_math_type);
    } else if (is_compatible_impl_enabled() && is_dim1_slice_non_contiguous(self) &&
               is_normal_broadcast_expanded(mat2)) {
        // aclnnMatmul internally optimizes away the dim-1 slice on self,
        // avoiding the extra copy that aclnnBatchMatMul would require.
        auto mat2_input = mat2[0];
        EXEC_NPU_CMD(aclnnMatmul, self, mat2_input, result, cube_math_type);
    } else {
        auto [self_input, mat2_input] = maybe_restore_broadcast(self, mat2);
        EXEC_NPU_CMD(aclnnBatchMatMul, self_input, mat2_input, result, cube_math_type);
    }

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    FLOP_COUNT(FlopCounter::bmm_flop, self, mat2);
    return result;
}

at::Tensor bmm(const at::Tensor &self, const at::Tensor &mat2)
{
    TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3D tensor");

    DO_MATMUL_COMPATIBILITY(aclnnBatchMatMulWeightNz, aclnnBatchMatMul, self, mat2, acl_op::bmm(self, mat2));

    // calculate the output size
    auto output_size = {self.size(0), self.size(1), mat2.size(2)};

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());

    int8_t cube_math_type = op_plugin::utils::get_cube_math_type_with_passthrough();
    if (op_plugin::utils::is_nz_format(mat2) && !op_plugin::utils::is_nz_format(self)) {
        EXEC_NPU_CMD(aclnnBatchMatMulWeightNz, self, mat2, result, cube_math_type);
    } else if (is_compatible_impl_enabled() && is_dim1_slice_non_contiguous(self) &&
               is_normal_broadcast_expanded(mat2)) {
        // aclnnMatmul internally optimizes away the dim-1 slice on self,
        // avoiding the extra copy that aclnnBatchMatMul would require.
        auto mat2_input = mat2[0];
        EXEC_NPU_CMD(aclnnMatmul, self, mat2_input, result, cube_math_type);
    } else {
        auto [self_input, mat2_input] = maybe_restore_broadcast(self, mat2);
        EXEC_NPU_CMD(aclnnBatchMatMul, self_input, mat2_input, result, cube_math_type);
    }

    auto outnames = at::namedinference::compute_bmm_outnames(result, self, mat2);
    at::namedinference::propagate_names_if_nonempty(result, outnames);
    FLOP_COUNT(FlopCounter::bmm_flop, self, mat2);
    return result;
}

}
