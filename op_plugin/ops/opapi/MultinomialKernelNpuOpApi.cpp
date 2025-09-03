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
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {

at::Tensor& multinomial_op_api(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator)
{
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(10);
        const uint64_t seed = pair.first;
        const uint64_t offset = pair.second;
        EXEC_NPU_CMD(aclnnMultinomial, self, num_samples, replacement, seed, offset, result);
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(10);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnMultinomialTensor, self, num_samples, replacement, *seed_ptr, *offset_ptr, offset_intragraph, result);
#endif
    }
    return result;
}

at::Tensor& multinomial_out(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator,
    at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnMultinomial, acl_op::multinomial_out(self, num_samples, replacement, generator, result));
    auto input_dim = self.dim();
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    output_size[input_dim - 1] = num_samples;
    at_npu::native::OpPreparation::check_tensor(
        {self},
        result,
        at::ScalarType::Long,
        output_size);
    multinomial_op_api(result, self, num_samples, replacement, generator);
    return result;
}

at::Tensor multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnMultinomial, acl_op::multinomial(self, num_samples, replacement, generator));
    auto dim = self.dim();
    auto shape = op_infer::array_to_small_vector(self.sizes());
    shape[dim-1] = num_samples;
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(shape, self.options().dtype(at::kLong));
    multinomial_op_api(result, self, num_samples, replacement, generator);
    return result;
}

}
