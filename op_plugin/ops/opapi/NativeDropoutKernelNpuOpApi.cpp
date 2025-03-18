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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;

std::tuple<at::Tensor, at::Tensor> _npu_dropout(const at::Tensor& self, double p)
{
    DO_COMPATIBILITY(aclnnDropoutGenMaskV2, acl_op::_npu_dropout(self, p));
    DO_COMPATIBILITY(aclnnDropoutDoMask, acl_op::_npu_dropout(self, p));

    int64_t length = (self.numel() + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self);
    at::Tensor mask;

    auto original_stream = c10_npu::getCurrentNPUStream();
    {
        // During the life cycle of this raii instance, the calcu stream is set as the
        // secondary stream, and tasks are distributed to the secondary stream. At the
        // same time, according to the one-stream-one-pool principle, memory is also
        // alloced from the pool of the secondary stream.
        c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
        mask = at_npu::native::OpPreparation::apply_tensor_without_format({length}, self.options().dtype(at::kByte));
        at::IntArrayRef shapeArray(self.sizes());

        // DropOutGenMask use seed and seed1 to generator a seed, like this:
        //  seed1   seed
        // 127~64   63~0
        // so, we set seed1 = 0 to ensure the seed which user set is equal to the seed
        // used by the operator DropOutGenMask
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        // At present, the default value of random number may be very large,
        // which will cause overflow in graph mode, so we set seed = 0 to avoid it.
        const uint64_t seed = pair.first;
        const uint64_t offset = pair.second;
        aclDataType dataType = at_npu::native::OpPreparation::convert_to_acl_data_type(self.scalar_type());
        EXEC_NPU_CMD(aclnnDropoutGenMaskV2, shapeArray, p, seed, offset, dataType, mask);
    }
    // When tasks on multiple streams read and write the same block of memory,
    // recordStream needs to be called to ensure the correctness of memory reuse.
    c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);

    EXEC_NPU_CMD(aclnnDropoutDoMask, self, mask, p, result);
    return std::tie(result, mask);
}

at::Tensor npu_dropout_backward(const at::Tensor& grad_output, const at::Tensor& mask, double scale)
{
    DO_COMPATIBILITY(aclnnDropoutDoMask, acl_op::npu_dropout_backward(grad_output, mask, scale));

    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(grad_output);
    EXEC_NPU_CMD(aclnnDropoutDoMask, grad_output, mask, scale, result);
    return result;
}

std::tuple<at::Tensor, at::Tensor> native_dropout(const at::Tensor& input, double p, c10::optional<bool> train)
{
    DO_COMPATIBILITY(aclnnDropoutGenMaskV2, acl_op::native_dropout(input, p, train));
    DO_COMPATIBILITY(aclnnDropoutDoMask, acl_op::native_dropout(input, p, train));

    bool dropout_train = !train.has_value() ? true : train.value();
    at::TensorOptions options = input.options();
    if (p == 0 || !dropout_train) {
        at::Tensor mask = at::ones(input.sizes(), options);
        return std::make_tuple(input.clone(), mask);
    }
    if (p == 1) {
        at::Tensor output = at::zeros(input.sizes(), options);
        at::Tensor mask = at::zeros(input.sizes(), options);
        return std::make_tuple(output, mask);
    }
    return op_api::_npu_dropout(input, p);
}

at::Tensor native_dropout_backward(const at::Tensor& grad_output, const at::Tensor& mask, double scale)
{
    DO_COMPATIBILITY(aclnnDropoutDoMask, acl_op::native_dropout_backward(grad_output, mask, scale));

    double p = (scale == 0.0) ? 1 : (1 - 1 / scale);
    if (p == 0) {
        return grad_output;
    }
    if (p == 1) {
        at::TensorOptions options = grad_output.options();
        return at::zeros(grad_output.sizes(), options);
    }
    return op_api::npu_dropout_backward(grad_output, mask, p);
}

}
