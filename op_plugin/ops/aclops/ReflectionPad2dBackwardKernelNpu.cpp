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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
bool check_padding(at::IntArrayRef padding)
{
    for (uint64_t i = 0; i < padding.size(); i++) {
        if (padding[i] != 0) {
            return false;
        }
    }
    return true;
}

at::Tensor &reflection_pad2d_backward_out_npu_nocheck(const at::Tensor &grad_output, const at::Tensor &input,
                                                      at::IntArrayRef padding, at::Tensor &grad_input)
{
    c10::SmallVector<int64_t, N> vector_int;
    c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(padding);
    at::Tensor input_cp = input;
    at::Tensor grad_output_cp = grad_output;
    if (input.dim() == 3) {
        input_cp = input.unsqueeze(0);
        grad_output_cp = grad_output.unsqueeze(0);
        grad_input.unsqueeze_(0);
    }
    TORCH_CHECK(input_cp.dim() != 0, "The input should not be empty" + OPS_ERROR(ErrCode::PARAM));
    paddings_vector.resize(2 * input_cp.dim(), 0);
    for (int64_t i = static_cast<int>(paddings_vector.size()); i > 0; i -= 2) {
        vector_int.emplace_back(paddings_vector[i - 2]);
        vector_int.emplace_back(paddings_vector[i - 1]);
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("PadV3Grad")
        .Input(grad_output_cp)
        .Input(vector_int, at::kInt)
        .Output(grad_input)
        .Attr("mode", static_cast<string>("reflect"))
        .Attr("paddings_contiguous", true)
        .Run();

    if (input.dim() == 3) {
        grad_input.squeeze_(0);
    }
    return grad_input;
}
} // namespace

at::Tensor &reflection_pad2d_backward_out(const at::Tensor &grad_output, const at::Tensor &self,
                                          at::IntArrayRef padding, at::Tensor &grad_input)
{
    if (check_padding(padding)) {
        grad_input.copy_(grad_output);
        return grad_input;
    }

    npu_preparation::CheckOut({self, grad_output}, grad_input, self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        reflection_pad2d_backward_out_npu_nocheck(grad_output, self, padding, contiguous_result);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        reflection_pad2d_backward_out_npu_nocheck(grad_output, self, padding, grad_input);
    }
    return grad_input;
}

at::Tensor reflection_pad2d_backward(const at::Tensor &grad_output, const at::Tensor &self, at::IntArrayRef padding)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    if (check_padding(padding)) {
        grad_input.copy_(grad_output);
        return grad_input;
    }

    reflection_pad2d_backward_out_npu_nocheck(grad_output, self, padding, grad_input);
    return grad_input;
}

} // namespace acl_op
