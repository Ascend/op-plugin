// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t MATMUL_ABFT_VERIFY_DIM = 2;
constexpr int64_t MATMUL_ABFT_VERIFY_ROW_PACK = 8;  // 8 row-segment verdicts packed per byte
constexpr int64_t MATMUL_ABFT_VERIFY_SPLIT_N = 256; // column split granularity

c10::SmallVector<int64_t, SIZE> matmul_abft_verify_out_size(const at::Tensor& a, const at::Tensor& b)
{
    // compRow shape: [ceil(M / 8) * splitN], where splitN = ceil(N / 256).
    int64_t row_bytes = op_infer::CeilDiv(a.size(0), MATMUL_ABFT_VERIFY_ROW_PACK);
    int64_t split_n = op_infer::CeilDiv(b.size(1), MATMUL_ABFT_VERIFY_SPLIT_N);
    c10::SmallVector<int64_t, SIZE> output_shape;
    output_shape.emplace_back(row_bytes * split_n);
    return output_shape;
}
} // namespace

at::Tensor _npu_matmul_abft_verify(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& checksum_weight,
    double e_max)
{
    TORCH_CHECK(a.dim() == MATMUL_ABFT_VERIFY_DIM, "a must be a 2D tensor, but got ", a.dim(),
        "D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(b.dim() == MATMUL_ABFT_VERIFY_DIM, "b must be a 2D tensor, but got ", b.dim(),
        "D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(c.dim() == MATMUL_ABFT_VERIFY_DIM, "c must be a 2D tensor, but got ", c.dim(),
        "D.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(checksum_weight.dim() == 1, "checksum_weight must be a 1D tensor, but got ",
        checksum_weight.dim(), "D.", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(a.size(1) == b.size(0), "The K dim of a (", a.size(1),
        ") must be equal to the K dim of b (", b.size(0), ").", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1),
        "c must have shape [M, N] = [", a.size(0), ", ", b.size(1), "] derived from a and b, but got [",
        c.size(0), ", ", c.size(1), "].", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(checksum_weight.size(0) == b.size(1), "checksum_weight must have shape [N] = [",
        b.size(1), "], but got [", checksum_weight.size(0), "].", OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(c.scalar_type() == at::ScalarType::Float,
        "c only supports float32, but got ", c.scalar_type(), ".", OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == checksum_weight.scalar_type(),
        "a, b and checksum_weight must have the same dtype, but got ", a.scalar_type(), ", ",
        b.scalar_type(), ", ", checksum_weight.scalar_type(), ".",
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half || a.scalar_type() == at::ScalarType::BFloat16 ||
        a.scalar_type() == at::ScalarType::Float,
        "a, b and checksum_weight only support float16/bfloat16/float32, but got ", a.scalar_type(),
        ".", OPS_ERROR(ErrCode::TYPE));

    TORCH_CHECK(e_max >= 0, "e_max must be non-negative, but got ", e_max, ".",
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous() &&
        checksum_weight.is_contiguous(),
        "a, b, c and checksum_weight only support contiguous tensors.",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(a) &&
        at_npu::native::FormatHelper::IsBaseFormatType(b) &&
        at_npu::native::FormatHelper::IsBaseFormatType(c) &&
        at_npu::native::FormatHelper::IsBaseFormatType(checksum_weight),
        "a, b, c and checksum_weight only support ND format.",
        OPS_ERROR(ErrCode::PARAM));

    auto output_shape = matmul_abft_verify_out_size(a, b);
    at::Tensor comp_row = npu_preparation::apply_tensor_without_format(
        output_shape, a.options().dtype(at::kByte));

    EXEC_NPU_CMD(aclnnMatmulAbftVerify, a, b, c, checksum_weight, e_max, comp_row);
    return comp_row;
}
} // namespace op_api
