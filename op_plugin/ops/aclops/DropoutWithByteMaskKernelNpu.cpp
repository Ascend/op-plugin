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

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_utils = at_npu::native::NpuUtils;
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;

namespace {
at::Tensor& dropout_do_mask_with_byte_mask(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar prob)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("DropOutDoMaskV3")
        .Input(self)
        .Input(mask)
        .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Run();
    return result;
}

at::Tensor dropout_gen_byte_mask(const at::Tensor& self, at::Scalar prob)
{
    at::IntArrayRef self_shape = self.sizes();
    at::Tensor mask = npu_preparation::apply_tensor_with_format(
        self_shape,
        self.options().dtype(at::kByte),
        ACL_FORMAT_ND);
    at_npu::native::OpCommand cmd;
    // If either seed or seed2 are set to be non-zero, the random number generator
    // is seeded by the given seed. Otherwise, it is seeded by a random seed.
    // DropOutGenMaskV3 use seed and seed2 to generator a seed, like this:
    //  seed2   seed
    // 127~64   63~0
    // so, we set seed2 = 0 to ensure the seed which user set is equal to the seed
    // used by the operator DropOutGenMaskV3
    const auto gen = at_npu::detail::getDefaultNPUGenerator();
    const int64_t seed = static_cast<int64_t>(gen.current_seed());
    const int64_t seed2 = 0;
    cmd.Name("DropOutGenMaskV3")
        .Input(self_shape)
        .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(mask)
        .Attr("seed", seed)
        .Attr("seed2", seed2)
        .Run();
    return mask;
}

std::tuple<at::Tensor, at::Tensor> dropout_out_nocheck(
    at::Tensor result,
    const at::Tensor& self,
    double p)
{
    at::Tensor self_cp = npu_utils::format_contiguous(self);
    TORCH_CHECK(
        p >= 0 && p <= 1,
        "dropout probability has to be between 0 and 1, but got ", p,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        at::isFloatingType(self_cp.scalar_type()),
        "dropout only supports floating-point dtypes" + OPS_ERROR(ErrCode::TYPE));

    double retain = 1. - p;
    at::Scalar prob = at::Scalar(retain);
    at::Tensor mask;
    auto original_stream = c10_npu::getCurrentNPUStream();
    {
        // During the life cycle of this raii instance, the calcu stream is set as the
        // secondary stream, and tasks are distributed to the secondary stream. At the
        // same time, according to the one-stream-one-pool principle, memory is also
        // alloced from the pool of the secondary stream.
        c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
        mask = dropout_gen_byte_mask(self_cp, prob);
    }
    // When tasks on multiple streams read and write the same block of memory,
    // recordStream needs to be called to ensure the correctness of memory reuse.
    c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
    dropout_do_mask_with_byte_mask(result, self_cp, mask, prob);

    return std::tie(result, mask);
}
} // namespace

at::Tensor _dropout_with_byte_mask_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double p)
{
    TORCH_CHECK(
        at::isFloatingType(grad_output.scalar_type()),
        "dropoutbackward only supports floating-point dtypes" + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        mask.scalar_type() == at::ScalarType::Byte,
        "mask should be torch.uint8 dtype" + OPS_ERROR(ErrCode::TYPE));
    double retain = 1. - p;
    at::Tensor result = npu_preparation::apply_tensor(grad_output);

    at_npu::native::OpCommand cmd;
    cmd.Name("DropOutDoMaskV3")
        .Input(grad_output)
        .Input(mask)
        .Input(at::Scalar(retain), grad_output.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Run();

    return result;
}

std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask(
    const at::Tensor& self,
    double p)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    return dropout_out_nocheck(result, self, p);
}

at::Tensor dropout_with_byte_mask(const at::Tensor& self, double p, bool train)
{
    TORCH_CHECK(
        torch_npu::utils::is_npu(self),
        "dropout_with_byte_mask only supports device for NPU!" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    if (p == 0 || !train || self.numel() == 0) {
        return self;
    }
    if (p == 1) {
        return self.mul(at::zeros(self.sizes(), self.options()));
    }
    auto results = at_npu::native::custom_ops::_dropout_with_byte_mask(self, p);
    return std::get<0>(results);
}

} // namespace acl_op
