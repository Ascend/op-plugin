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


#include <ATen/ops/_empty_affine_quantized.h>
#include <torch/library.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
namespace {
#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
// Currently we only support int8 symmetric (zero_point = 0 for inputs and output) quantized add.
// We implement relu ( (qa + qb * ( b_scale / a_scale) ) ) * ( a_scale / out_scale )
// which requires 2 multiplication, 1 addition, and 1 relu ops.
template <bool kReluFused = false>
at::Tensor add(at::Tensor qa, at::Tensor qb, double output_scale, int64_t output_zero_point)
{
    if (qa.numel() == 0) {
        return at::Tensor{};
    }

    TORCH_CHECK(
        output_scale != 0, "output_scale can not be zero");
    // For now we just assume the input tensors are the same shape and don't consider broadcasted add.
    TORCH_CHECK(
        qa.sizes() == qb.sizes(),
        "Quantized npu add currently expects both input tensors to be the same shape");
    TORCH_CHECK(
        qa.qscheme() == c10::kPerTensorAffine,
        "Only per tensor quantization is supported in Add.");
    TORCH_CHECK(
        qa.qscheme() == qb.qscheme(),
        "Both inputs to Add must have the same quantization scheme.");
    TORCH_CHECK(
        qa.scalar_type() == qb.scalar_type(),
        "Add operands should have same data type.");
    TORCH_CHECK(
        qa.scalar_type() == at::ScalarType::QInt8,
        "Add operands expect scalar type QInt8");

    at::Tensor qa_float = qa.int_repr().to(at::kFloat);
    at::Tensor qb_float = qb.int_repr().to(at::kFloat);
    at::Tensor calc_tensor = at::empty(
        qa.sizes(),
        qa.options().dtype(at::ScalarType::Float).memory_format(qa.suggest_memory_format()));

    // computes relu ( (qa + qb * ( b_scale / a_scale) ) ) * ( a_scale / out_scale )
    calc_tensor = qa_float + qb_float * (qb.q_scale() / qa.q_scale());
    if (kReluFused) {
        calc_tensor = at::relu(calc_tensor);
    }
    calc_tensor = calc_tensor * (qa.q_scale() / output_scale);
    calc_tensor = at::clamp(calc_tensor, -128.0, 127.0);

    at::Tensor tmp_result = calc_tensor.round().to(at::kChar);
    at::Tensor result = at::_empty_affine_quantized(qa.sizes(), qa.options().dtype(at::ScalarType::QInt8),
                                                    output_scale, output_zero_point, qa.suggest_memory_format());
    at_npu::native::NPUNativeFunctions::set_(result, tmp_result);
    return result;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
    m.impl("quantized::add", TORCH_FN(add<false>));
    m.impl("quantized::add_relu", TORCH_FN(add<true>));
}
#endif
} // namespace
} // namespace acl_op
