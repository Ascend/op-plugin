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
#include "op_plugin/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;
using npu_utils = at_npu::native::NpuUtils;

namespace {
const static int64_t LENGTH_DATA_ALIGN = 128;
at::Tensor dropout_genmask(const at::Tensor &self, at::Scalar prob)
{
    uint32_t length = (static_cast<uint32_t>(self.numel()) + LENGTH_DATA_ALIGN - 1) / LENGTH_DATA_ALIGN * LENGTH_DATA_ALIGN;
    at::Tensor mask =
        npu_preparation::apply_tensor_with_format({length}, self.options().dtype(at::kByte), ACL_FORMAT_ND);
    at::IntArrayRef self_shape = self.sizes();

    int64_t seed = 2;
    int64_t seed2 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("DropOutGenMaskV3")
        .Input(self_shape)
        .Input(prob, self.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(mask)
        .Attr("seed", seed)
        .Attr("seed2", seed2)
        .Run();
    return mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_with_add_softmax_forward(const at::Tensor &self,
                                                                                    const at::Tensor &x1,
                                                                                    at::Scalar alpha, double p,
                                                                                    int64_t dim)
{
    at::Tensor result_softmax = npu_preparation::apply_tensor(x1);
    at::Tensor result_dropout = npu_preparation::apply_tensor(self);
    c10::SmallVector<int64_t, N> dimList = {dim};
    double retain = 1. - p;
    at::Scalar prob = at::Scalar(retain);
    at::Tensor mask;
    auto original_stream = c10_npu::getCurrentNPUStream();
    {
        c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
        mask = dropout_genmask(x1, prob);
    }
    c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);

    at_npu::native::OpCommand cmd;
    cmd.Name("AxpyWithSoftmaxAndDropOutDoMask")
        .Input(x1)
        .Input(self)
        .Input(mask)
        .Output(result_softmax)
        .Output(result_dropout)
        .Attr("alpha", alpha)
        .Attr("input_keep_prob", prob)
        .Attr("axis", dimList)
        .Run();
    return std::tie(mask, result_softmax, result_dropout);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor &grad,
                                                                         const at::Tensor &mask,
                                                                         const at::Tensor &softmax_out,
                                                                         const at::Scalar &alpha, double prob, int64_t dim)
{
    at::Tensor result = npu_preparation::apply_tensor(softmax_out);
    c10::SmallVector<int64_t, N> dimList = {dim};
    double retain = 1. - prob;
    at::Scalar p = at::Scalar(retain);

    at_npu::native::OpCommand cmd;
    cmd.Name("DropoutWithMulsAndSoftmaxGrad")
        .Input(grad)
        .Input(mask)
        .Input(softmax_out)
        .Output(result)
        .Attr("alpha", alpha)
        .Attr("input_keep_prob", p)
        .Attr("axes", dimList)
        .Run();
    return std::tie(result, grad);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_with_add_softmax(const at::Tensor &self,
                                                                            const at::Tensor &x1,
                                                                            const at::Scalar &alpha,
                                                                            double prob,
                                                                            int64_t dim)
{
    return npu_dropout_with_add_softmax_forward(self, x1, alpha, prob, dim);
}
} // namespace acl_op
