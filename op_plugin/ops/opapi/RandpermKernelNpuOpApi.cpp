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
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& randperm_op_api(int64_t n, c10::optional<at::Generator> gen_, at::Tensor& result)
{
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    EXEC_NPU_CMD(aclnnRandperm, n, pair.first, pair.second, result);
    return result;
}

at::Tensor& randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnRandperm, acl_op::randperm_out(n, generator, out));
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n, OPS_ERROR(ErrCode::VALUE));
    at_npu::native::OpPreparation::check_tensor({}, out, out, {n});
    randperm_op_api(n, generator, out);
    return out;
}

at::Tensor& randperm_out(int64_t n, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnRandperm, acl_op::randperm_out(n, out));
    at_npu::native::OpPreparation::check_tensor({}, out, out, {n});
    c10::optional<at::Generator> generator = static_cast<c10::optional<at::Generator>>(c10::nullopt);
    randperm_op_api(n, generator, out);
    return out;
}

at::Tensor randperm(int64_t n, c10::optional<at::Generator> generator,
                    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                    c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnRandperm, acl_op::randperm(n, generator, dtype, layout, device, pin_memory));
    TORCH_CHECK(n >= 0, "n must be non-negative, got", n, OPS_ERROR(ErrCode::VALUE));
    at::TensorOptions options = options.dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format({n}, options);

    randperm_op_api(n, generator, result);
    return result;
}

at::Tensor randperm(int64_t n, c10::optional<at::ScalarType> dtype,
                    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                    c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnRandperm, acl_op::randperm(n, dtype, layout, device, pin_memory));
    return op_api::randperm(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), dtype, layout,
                            device, pin_memory);
}

}
