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

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_format_helper = at_npu::native::FormatHelper;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor linear_npu_nocheck(const at::Tensor &input, const at::Tensor &weight,
                              const c10::optional<at::Tensor> &bias_opt)
{
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    c10::SmallVector<int64_t, SIZE> output_size = {input.size(0), weight.size(0)};
    at::Tensor output = npu_preparation::apply_tensor(input, output_size);

    int64_t offset_x = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("MatMulV2").Input(input).Input(weight);
    if (bias.defined()) {
        cmd.Input(bias);
    }
    cmd.Output(output).Attr("transpose_x1", false).Attr("transpose_x2", true).Attr("offset_x", offset_x).Run();

    return output;
}

at::Tensor linear_backward_out_npu_nocheck(at::Tensor &result, const at::Tensor &input, const at::Tensor &weight,
                                           bool transpose_x1, bool transpose_x2)
{
    int64_t offset_x = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("MatMulV2")
        .Input(input)
        .Input(weight)
        .Output(result)
        .Attr("transpose_x1", transpose_x1)
        .Attr("transpose_x2", transpose_x2)
        .Attr("offset_x", offset_x)
        .Run();
    return result;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_linear_backward(const at::Tensor &grad, const at::Tensor &input,
                                                       const at::Tensor &weight)
{
    TORCH_CHECK(grad.dim() >= 2, "torch.nn.functional.linear() grad must be at least two-dimensional."
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.dim() >= 2, "torch.nn.functional.linear() input must be at least two-dimensional."
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 2, "torch.nn.functional.linear() weight must be at least two-dimensional."
        + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> input_grad_output_size = {grad.size(0), weight.size(1)};
    c10::SmallVector<int64_t, SIZE> weight_grad_output_size = {grad.size(1), input.size(1)};
    at::Tensor input_grad = npu_preparation::apply_tensor(input, input_grad_output_size);
    at::Tensor weight_grad = npu_preparation::apply_tensor(weight, weight_grad_output_size);

    if (npu_preparation::get_tensor_npu_format(grad) == npu_preparation::get_tensor_npu_format(weight)) {
        linear_backward_out_npu_nocheck(input_grad, grad, weight, false, false);
        linear_backward_out_npu_nocheck(weight_grad, grad, input, true, false);
    } else {
        at::Tensor gradFormatcast = npu_preparation::apply_tensor(grad, grad.sizes());
        gradFormatcast =
            at_npu::native::custom_ops::npu_format_cast(grad, npu_preparation::get_tensor_npu_format(weight));
        linear_backward_out_npu_nocheck(input_grad, gradFormatcast, weight, false, false);
        linear_backward_out_npu_nocheck(weight_grad, gradFormatcast, input, true, false);
    }

    return std::tie(input_grad, weight_grad);
}

at::Tensor npu_linear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias)
{
    TORCH_CHECK(input.dim() >= 2, "torch.nn.functional.linear() input must be at least two-dimensional."
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 2, "torch.nn.functional.linear() weight must be at least two-dimensional."
        + OPS_ERROR(ErrCode::PARAM));

    auto is_aligin = [&]() {
        return ((static_cast<uint64_t>(input.size(0)) & 0x0000000F) == 0) &&
               ((static_cast<uint64_t>(input.size(1)) & 0x0000000F) == 0) &&
               ((static_cast<uint64_t>(weight.size(0)) & 0x0000000F) == 0) &&
               ((static_cast<uint64_t>(weight.size(1)) & 0x0000000F) == 0);
    };

    static auto mm_bmm_nd = !at_npu::native::env::CheckMmBmmNDDisable();
    static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
    at::Tensor input_cast = (npu_format_helper::IsBaseFormatType(input) && mm_bmm_nd &&
                             ((is_support_nd_out && op_plugin::utils::is_nd_to_nz_on_fly(input, weight)) ||
                              (!is_support_nd_out && is_aligin()))) ?
                                input :
                                at_npu::native::custom_ops::npu_format_cast(input, ACL_FORMAT_FRACTAL_NZ);
    return linear_npu_nocheck(input_cast, weight, bias);
}
} // namespace acl_op
