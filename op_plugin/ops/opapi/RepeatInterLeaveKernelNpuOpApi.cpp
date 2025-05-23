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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const static int INT64T_SIZE = 8;

// convert dim to non-negative value
int64_t wrap_dim(const at::Tensor &self, c10::optional<int64_t> dim)
{
    int64_t real_dim = dim.value_or(0);
    return (real_dim < 0) ? (real_dim + self.dim()) : real_dim;
}

// check tensor repeats is valid
bool check_tensor_repeats(const at::Tensor &self, const at::Tensor &repeats, c10::optional<int64_t> dim)
{
    if (repeats.dim() == 0) {
        return true;
    }

    if (repeats.dim() == 1) {
        if (dim.has_value()) {
        // with dim check repeats is rank 1 with 1 element / rank 1 with (self.size(dim)) elements
            int64_t real_dim = wrap_dim(self, dim);
            if (repeats.size(0) == self.size(real_dim) || repeats.size(0) == 1) {
                return true;
            }
        } else {
            // without dim: check repeats is rank 0/ rank 1 with 1 element / rank 1 with (self.numel()) elements
            if (repeats.size(0) == self.numel() || repeats.size(0) == 1) {
                return true;
            }
        }
    }

    return false;
}

// check dim is in range [-self.dim(), self.dim()-1]
bool check_dim_valid(const at::Tensor &self, c10::optional<int64_t> dim)
{
    int64_t real_dim = dim.value_or(0);
    int64_t self_dim = self.dim();
    int64_t dim_min = std::min(-self_dim, self_dim - 1);
    int64_t dim_max = std::max(-self_dim, self_dim - 1);
    return (dim_min <= real_dim && real_dim <= dim_max);
}

at::Tensor apply_result_tensor(const at::Tensor &self, c10::SmallVector<int64_t, INT64T_SIZE> &output_shape,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    if (output_size.has_value() && self.numel() != 0) {
        TORCH_CHECK(output_size_expected == output_size, "Allocated size does not match required size.", OPS_ERROR(ErrCode::VALUE));
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_shape);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor repeat_interleave(const at::Tensor& self, int64_t repeats,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveIntWithDim, acl_op::repeat_interleave(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleaveInt, acl_op::repeat_interleave(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range.", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(repeats >= 0, "repeats can not be negative.", OPS_ERROR(ErrCode::PARAM));

    // check output_size value is valid
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats, dim);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveIntWithDim, self, repeats, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleaveInt, self, repeats, output_size_expected, result);
    }

    return result;
}

at::Tensor repeat_interleave(const at::Tensor& self, const at::Tensor& repeats,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveWithDim, acl_op::repeat_interleave(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleave, acl_op::repeat_interleave(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range.", OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(check_tensor_repeats(self, repeats, dim), "repeats must have the same size as input along dim", OPS_ERROR(ErrCode::PARAM));

    // check output_size value is valid
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats, dim, output_size);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveWithDim, self, repeats, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleave, self, repeats, output_size_expected, result);
    }

    return result;
}
#endif

#if VERSION_BETWEEN(V2R1, V2R1)
at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size)
{
    int64_t repeats_int = repeats.expect_int();
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveIntWithDim,
                         acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleaveInt,
                         acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range." + OPS_ERROR(ErrCode::VALUE))
    TORCH_CHECK(repeats_int >= 0, "repeats can not be negative." + OPS_ERROR(ErrCode::VALUE));

    // check output_size value is valid
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats_int, dim);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveIntWithDim, self, repeats_int, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleaveInt, self, repeats_int, output_size_expected, result);
    }
    return result;
}

at::Tensor repeat_interleave(const at::Tensor& self, const at::Tensor& repeats,
    c10::optional<int64_t> dim, c10::optional<int64_t> output_size)
{
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveWithDim, acl_op::repeat_interleave(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleave, acl_op::repeat_interleave(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range." + OPS_ERROR(ErrCode::VALUE))
    TORCH_CHECK(check_tensor_repeats(self, repeats, dim), "repeats must have the same size as input along dim" + OPS_ERROR(ErrCode::PARAM));

    // check output_size value is valid
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats, dim, output_size);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveWithDim, self, repeats, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleave, self, repeats, output_size_expected, result);
    }
    return result;
}
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<c10::SymInt> output_size)
{
    int64_t repeats_int = repeats.expect_int();
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveIntWithDim,
                         acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleaveInt,
                         acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range." + OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(repeats_int >= 0, "repeats can not be negative." + OPS_ERROR(ErrCode::PARAM));

    // check output_size value is valid
    c10::optional<int64_t> _output_size = c10::nullopt;
    if (output_size.has_value()) {
        int64_t output_size_val = output_size.value().expect_int();
        _output_size = c10::optional<int64_t>(output_size_val);
    }
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats_int, dim);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, _output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveIntWithDim, self, repeats_int, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleaveInt, self, repeats_int, output_size_expected, result);
    }
    return result;
}

at::Tensor repeat_interleave_symint(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<c10::SymInt> output_size)
{
    if (dim.has_value()) {
        DO_COMPATIBILITY(aclnnRepeatInterleaveWithDim, acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    } else {
        DO_COMPATIBILITY(aclnnRepeatInterleave, acl_op::repeat_interleave_symint(self, repeats, dim, output_size));
    }

    // argument repeat and dim must be valid
    TORCH_CHECK(check_dim_valid(self, dim), "dim value is not in valid range." + OPS_ERROR(ErrCode::PARAM))
    TORCH_CHECK(check_tensor_repeats(self, repeats, dim), "repeats must have the same size as input along dim" + OPS_ERROR(ErrCode::PARAM));

    // check output_size value is valid
    c10::optional<int64_t> _output_size = c10::nullopt;
    if (output_size.has_value()) {
        int64_t output_size_val = output_size.value().expect_int();
        _output_size = c10::optional<int64_t>(output_size_val);
    }
    auto output_shape = op_infer::repeat_interleave_npu_output_size_opapi(self, repeats, dim, _output_size);
    int64_t cur_dim = wrap_dim(self, dim);
    int64_t output_size_expected = output_shape[cur_dim];
    at::Tensor result = apply_result_tensor(self, output_shape, dim, _output_size);

    if (dim.has_value()) {
        int64_t real_dim = dim.value_or(0);
        EXEC_NPU_CMD(aclnnRepeatInterleaveWithDim, self, repeats, real_dim, output_size_expected, result);
    } else {
        EXEC_NPU_CMD(aclnnRepeatInterleave, self, repeats, output_size_expected, result);
    }
    return result;
}
#endif
}
