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
using npu_compile_type = at_npu::native::CompileType;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& normal_out_npu_nocheck(at::Tensor& result, c10::optional<at::Generator> gen)
{
    auto gen_default =
        at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen_default->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);

    at::SmallVector<int64_t, N> key = {seed};
    at::SmallVector<int64_t, N> counter = {0, offset};
    const int32_t alg = 1;

    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessRandomNormalV2")
        .Input(result.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(key, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
        .Input(counter, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
        .Input(at::Scalar(alg), at::ScalarType::Int)
        .Output(result)
        .Attr("dtype", result.scalar_type())
        .Run();
    return result;
}
} // namespace

at::Tensor& normal_out(
    const at::Tensor& mean,
    double std,
    c10::optional<at::Generator> generator,
    at::Tensor& result)
{
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std,
        OPS_ERROR(ErrCode::VALUE));

    npu_preparation::CheckOut({mean}, result, mean);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        normal_out_npu_nocheck(contiguous_result, generator);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        normal_out_npu_nocheck(result, generator);
    }
    result.mul_(std).add_(mean);
    return result;
}

at::Tensor& normal_out(
    double mean,
    const at::Tensor& std,
    c10::optional<at::Generator> generator,
    at::Tensor& result)
{
    npu_preparation::CheckOut({std}, result, std);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        normal_out_npu_nocheck(contiguous_result, generator);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        normal_out_npu_nocheck(result, generator);
    }

    result.mul_(std).add_(mean);
    return result;
}

at::Tensor& normal_out(
    const at::Tensor& mean,
    const at::Tensor& std,
    c10::optional<at::Generator> generator,
    at::Tensor& result)
{
    at::SmallVector<int64_t, SIZE> output_size = op_infer::broadcast_ops_npu_output_size(mean, std);
    npu_preparation::CheckOut({mean, std}, result, mean, output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        normal_out_npu_nocheck(contiguous_result, generator);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        normal_out_npu_nocheck(result, generator);
    }

    result.mul_(std).add_(mean);
    return result;
}

at::Tensor& normal_out(
    double mean,
    double std,
    at::IntArrayRef size,
    c10::optional<at::Generator> generator,
    at::Tensor& result)
{
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std,
        OPS_ERROR(ErrCode::VALUE));
    npu_preparation::CheckOut({}, result, result, size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        normal_out_npu_nocheck(contiguous_result, generator);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        normal_out_npu_nocheck(result, generator);
    }

    result.mul_(std).add_(mean);
    return result;
}

at::Tensor normal(
    const at::Tensor& mean,
    double std,
    c10::optional<at::Generator> generator)
{
    at::Tensor result = npu_preparation::apply_tensor(mean);
    normal_out_npu_nocheck(result, generator);
    result.mul_(std).add_(mean);
    return result;
}

at::Tensor normal(
    double mean,
    const at::Tensor& std,
    c10::optional<at::Generator> generator)
{
    at::Tensor result = npu_preparation::apply_tensor(std);
    normal_out_npu_nocheck(result, generator);
    result.mul_(std).add_(mean);
    return result;
}

at::Tensor normal(
    const at::Tensor& mean,
    const at::Tensor& std,
    c10::optional<at::Generator> generator)
{
    at::SmallVector<int64_t, SIZE> output_size = op_infer::broadcast_ops_npu_output_size(mean, std);
    at::Tensor result = npu_preparation::apply_tensor(mean, output_size);
    normal_out_npu_nocheck(result, generator);
    result.mul_(std).add_(mean);
    return result;
}

at::Tensor normal(
    double mean, double std,
    at::IntArrayRef size,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                    .device(device_opt)
                                                    .layout(layout_opt)
                                                    .pinned_memory(pin_memory_opt);
    at::Tensor result = npu_preparation::apply_tensor_with_format(size, option, ACL_FORMAT_ND);
    normal_out_npu_nocheck(result, generator);
    result.mul_(std).add_(mean);
    return result;
}

at::Tensor& normal_(
    at::Tensor& self,
    double mean,
    double std,
    c10::optional<at::Generator> generator)
{
    acl_op::normal_out(mean, std, self.sizes(), generator, self);
    return self;
}
} // namespace acl_op
