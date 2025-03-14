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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& multinomial_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen)
{
    auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen_->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);

    at_npu::native::OpCommand cmd;
    cmd.Name("MultinomialWithReplacement")
        .Input(self)
        .Input(at::Scalar(seed), at::ScalarType::Long)
        .Input(at::Scalar(offset), at::ScalarType::Long)
        .Output(result)
        .Attr("numsamples", num_samples)
        .Attr("replacement", replacement)
        .Run();
    return result;
}
} // namespace

at::Tensor& multinomial_out(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator,
    at::Tensor& out)
{
    auto input_dim = self.dim();
    TORCH_CHECK(input_dim == 1 || input_dim == 2, "dim of input tensor only can be 1 or 2."
                + OPS_ERROR(ErrCode::PARAM));

    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size[input_dim - 1] = num_samples;
    npu_preparation::CheckOut(
        {self},
        out,
        npu_preparation::get_tensor_npu_format(out),
        at::ScalarType::Long,
        output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        multinomial_out_npu_nocheck(contiguous_result, self, num_samples, replacement, generator);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        multinomial_out_npu_nocheck(out, self, num_samples, replacement, generator);
    }
    return out;
}

at::Tensor multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator)
{
    auto dim = self.dim();
    TORCH_CHECK(dim == 1 || dim == 2, "dim of input tensor only can be 1 or 2."
                + OPS_ERROR(ErrCode::PARAM));

    auto shape = op_infer::array_to_small_vector(self.sizes());
    shape[dim - 1] = num_samples;
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        shape,
        self.options().dtype(at::kLong),
        npu_preparation::get_tensor_npu_format(self));
    multinomial_out_npu_nocheck(result, self, num_samples, replacement, generator);
    return result;
}
} // namespace acl_op
