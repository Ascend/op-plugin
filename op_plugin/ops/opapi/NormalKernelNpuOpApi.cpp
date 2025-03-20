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
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnInplaceNormal, acl_op::normal_(self, mean, std, generator));
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std, OPS_ERROR(ErrCode::VALUE));
    npu_preparation::check_tensor({}, self, self, self.sizes());
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float mean_cast = static_cast<float>(mean);
    float rstd_cast = static_cast<float>(std);
    EXEC_NPU_CMD(aclnnInplaceNormal, self, mean_cast, rstd_cast, seed, offset);
    return self;
}

/* TensorTensor */
at::Tensor& normal_out(const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> generator,
                       at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnNormalTensorTensor, acl_op::normal_out(mean, std, generator, out));
    at::SmallVector<int64_t, SIZE> output_size = op_infer::broadcast_ops_npu_output_size(mean, std);
    npu_preparation::check_tensor({mean, std}, out, out, output_size);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    EXEC_NPU_CMD(aclnnNormalTensorTensor, mean, std, seed, offset, out);
    return out;
}

at::Tensor normal(const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnNormalTensorTensor, acl_op::normal(mean, std, generator));
    at::SmallVector<int64_t, SIZE> output_size = op_infer::broadcast_ops_npu_output_size(mean, std);
    at::Tensor result = npu_preparation::apply_tensor_without_format(mean, output_size);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    EXEC_NPU_CMD(aclnnNormalTensorTensor, mean, std, seed, offset, result);
    return result;
}

/* TensorFloat */
at::Tensor& normal_out(const at::Tensor& mean, double std, c10::optional<at::Generator> generator,
                       at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnNormalTensorFloat, acl_op::normal_out(mean, std, generator, out));
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std, OPS_ERROR(ErrCode::VALUE));
    npu_preparation::check_tensor({mean}, out, out);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float rstd_cast = static_cast<float>(std);
    EXEC_NPU_CMD(aclnnNormalTensorFloat, mean, rstd_cast, seed, offset, out);
    return out;
}

at::Tensor normal(const at::Tensor& mean, double std, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnNormalTensorFloat, acl_op::normal(mean, std, generator));
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std, OPS_ERROR(ErrCode::VALUE));
    at::Tensor result = npu_preparation::apply_tensor_without_format(mean);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float rstd_cast = static_cast<float>(std);
    EXEC_NPU_CMD(aclnnNormalTensorFloat, mean, rstd_cast, seed, offset, result);
    return result;
}

/* FloatTensor */
at::Tensor& normal_out(double mean, const at::Tensor& std, c10::optional<at::Generator> generator,
                       at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnNormalFloatTensor, acl_op::normal_out(mean, std, generator, out));
    npu_preparation::check_tensor({std}, out, out);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float mean_cast = static_cast<float>(mean);
    EXEC_NPU_CMD(aclnnNormalFloatTensor, mean_cast, std, seed, offset, out);
    return out;
}

at::Tensor normal(double mean, const at::Tensor& std, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnNormalFloatTensor, acl_op::normal(mean, std, generator));
    at::Tensor result = npu_preparation::apply_tensor_without_format(std);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float mean_cast = static_cast<float>(mean);
    EXEC_NPU_CMD(aclnnNormalFloatTensor, mean_cast, std, seed, offset, result);
    return result;
}

/* FloatFloat */
at::Tensor& normal_out(double mean, double std, at::IntArrayRef size,
                       c10::optional<at::Generator> generator, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnNormalFloatFloat, acl_op::normal_out(mean, std, size, generator, out));
    TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std, OPS_ERROR(ErrCode::VALUE));
    npu_preparation::check_tensor({}, out, out, size);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float mean_cast = static_cast<float>(mean);
    float rstd_cast = static_cast<float>(std);
    EXEC_NPU_CMD(aclnnNormalFloatFloat, mean_cast, rstd_cast, seed, offset, out);
    return out;
}

at::Tensor normal(double mean, double std,
                  at::IntArrayRef size,
                  c10::optional<at::Generator> generator,
                  c10::optional<at::ScalarType> dtype,
                  c10::optional<c10::Layout> layout,
                  c10::optional<c10::Device> device,
                  c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnNormalFloatFloat, acl_op::normal(mean, std, size, generator, dtype,
                                                           layout, device, pin_memory));
    c10::TensorOptions option = c10::TensorOptions().dtype(dtype)
        .device(device)
        .layout(layout)
        .pinned_memory(pin_memory);
    at::Tensor result = npu_preparation::apply_tensor_without_format(size, option);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    float mean_cast = static_cast<float>(mean);
    float rstd_cast = static_cast<float>(std);
    EXEC_NPU_CMD(aclnnNormalFloatFloat, mean_cast, rstd_cast, seed, offset, result);
    return result;
}

}
