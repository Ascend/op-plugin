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

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace acl_op {
using npu_format_helper = at_npu::native::FormatHelper;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&> batch_norm_backward_training_update_nocheck(
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps)
{
    at_npu::native::OpCommand cmd;

    string name = (self.dim() == 5) ? "BN3DTrainingUpdateGrad" : "BNTrainingUpdateGrad";
    cmd.Name(name)
        .Input(grad_out, "grads")
        .Input(self, "x")
        .Input(save_mean, "batch_mean")
        .Input(save_invstd, "batch_variance")
        .Output(grad_weight, "diff_scale")
        .Output(grad_bias, "diff_offset")
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    return std::tuple<at::Tensor&, at::Tensor&>(grad_weight, grad_bias);
}

at::Tensor& batch_norm_backward_training_reduce_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_weight,
    const at::Tensor& grad_bias,
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps)
{
    at_npu::native::OpCommand cmd;

    string name = (self.dim() == 5) ? "BN3DTrainingReduceGrad" : "BNTrainingReduceGrad";
    at::Tensor weight_cp = weight;
    auto self_format = npu_preparation::get_tensor_npu_format(self);
    auto weight_format = npu_preparation::get_tensor_npu_format(weight);

    bool check_bn_5hd = (self_format == ACL_FORMAT_NC1HWC0 && weight_format == ACL_FORMAT_ND) ? true : false;
    if (check_bn_5hd) {
        npu_format_helper::unsafe_format_cast(weight_cp, ACL_FORMAT_ND, ACL_FORMAT_NC1HWC0);
    }
    cmd.Name(name)
        .Input(grad_out, "grads")
        .Input(self, "x")
        .Input(grad_weight, "diff_scale")
        .Input(grad_bias, "diff_offset")
        .Input(weight_cp, "scale")
        .Input(save_mean, "batch_mean")
        .Input(save_invstd, "batch_variance")
        .Output(grad_input, "y")
        .Attr("epsilon", static_cast<float>(eps))
        .Run();
    if (check_bn_5hd) {
        npu_format_helper::unsafe_format_cast(weight_cp, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
    }
    return grad_input;
}

at::Tensor& batch_norm_backward_infer_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_weight,
    const at::Tensor& grad_bias,
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("BNInferGrad")
        .Input(grad_out, "grads")
        .Input(weight, "scale")
        .Input(running_var, "batch_variance")
        .Output(grad_input, "x_backprop")
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    return grad_input;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_backward_impl(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask)
{
    // note: when not train, save_mean/save_invstd replaced by running_mean/running_var
    at::Tensor mean = train ? save_mean : running_mean;
    at::Tensor invstd = train ? save_invstd : running_var;

    batch_norm_backward_training_update_nocheck(grad_weight, grad_bias, grad_out, self, weight, running_mean, running_var,
        mean, invstd, train, eps);

    if (grad_input_mask[0]) {
        if (!train) {
            batch_norm_backward_infer_nocheck(grad_input, grad_weight, grad_bias, grad_out, self, weight, running_mean,
                running_var, mean, invstd, train, eps);
        } else {
            batch_norm_backward_training_reduce_nocheck(grad_input, grad_weight, grad_bias, grad_out, self, weight,
                running_mean, running_var, mean, invstd, train, eps);
        }
    }

    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(grad_input, grad_weight, grad_bias);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask)
{
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] { return at::Tensor(); });
    const at::Tensor& save_mean = c10::value_or_else(save_mean_opt, [] { return at::Tensor(); });
    const at::Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] { return at::Tensor(); });

    at::Tensor self_reshape;
    at::Tensor grad_out_reshape;
    c10::SmallVector<int64_t, N> self_shape = op_infer::array_to_small_vector(self.sizes());

    if (grad_out.dim() <= 4) {
        c10::SmallVector<int64_t, N> nchw_shape(self_shape);
        nchw_shape.resize(4, 1);
        self_reshape = self.reshape(nchw_shape);
        grad_out_reshape = grad_out.reshape(nchw_shape);
    } else if (train && grad_out.dim() == 5) {
        // Use 3D BN ops for training, merging axes is not required.
        self_reshape = self;
        grad_out_reshape = grad_out;
    } else {
        // Infering uses 2dInfer Op, case no matched 3DInfer Op
        // ncdhw -> ndchw
        self_reshape = self.permute({0, 2, 1, 3, 4});
        grad_out_reshape = grad_out.permute({0, 2, 1, 3, 4});
        // nchw=(n*d, c, h, w)
        c10::SmallVector<int64_t, N> nchw_shape =
            {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
        // ndchw -> nchw
        self_reshape = self_reshape.reshape(nchw_shape);
        grad_out_reshape = grad_out_reshape.reshape(nchw_shape);
    }

    int64_t dim_c = self_reshape.size(1);
    at::TensorOptions options = self.options().dtype(at::ScalarType::Float);

    at::Tensor weight_cp = weight;
    at::Tensor running_mean_cp = running_mean;
    at::Tensor running_var_cp = running_var;

    at::Tensor weight_tensor = weight.defined() ? weight_cp : at::ones({dim_c}, options);
    at::Tensor running_mean_tensor = running_mean.defined() ? running_mean_cp : at::zeros({dim_c}, options);
    at::Tensor running_var_tensor = running_var.defined() ? running_var_cp : at::ones({dim_c}, options);

    at::Tensor grad_input = npu_preparation::apply_tensor(self_reshape.sizes(), self_reshape.options(), self_reshape);
    at::Tensor grad_weight = (grad_out.dim() == 5) ?
        npu_preparation::apply_tensor(weight_tensor, weight_tensor.options().dtype(at::ScalarType::Float)) :
        npu_preparation::apply_tensor(
            weight_tensor.sizes(), weight_tensor.options().dtype(at::ScalarType::Float), grad_out);
    at::Tensor grad_bias = (grad_out.dim() == 5) ?
        npu_preparation::apply_tensor(weight_tensor, weight_tensor.options().dtype(at::ScalarType::Float)) :
        npu_preparation::apply_tensor(
            weight_tensor.sizes(), weight_tensor.options().dtype(at::ScalarType::Float), grad_out);

    batch_norm_backward_impl(grad_input, grad_weight, grad_bias, grad_out_reshape, self_reshape, weight_tensor,
        running_mean_tensor, running_var_tensor, save_mean, save_invstd, train, eps, grad_input_mask);

    at::Tensor undefine_grad_input;
    at::Tensor undefine_grad_weight;
    at::Tensor undefine_grad_bias;

    if (grad_input_mask[0]) {
        if (!train && self.dim() == 5) {
            // NCHW -> NDCHW ->NCDHW
            std::swap(self_shape[1], self_shape[2]);
            grad_input = grad_input.view(self_shape);
            grad_input = npu_utils::format_contiguous(grad_input);
            grad_input = grad_input.permute({0, 2, 1, 3, 4}).clone();
        } else if (self.dim() < 5) {
            grad_input = grad_input.view(self_shape);
            grad_input = npu_utils::format_contiguous(grad_input);
        }
    } else {
        grad_input = undefine_grad_input;
    }

    if (!grad_input_mask[1]) {
        grad_weight = undefine_grad_weight;
    }

    if (!grad_input_mask[2]) {
        grad_bias = undefine_grad_bias;
    }

    if (grad_weight.defined()) {
        auto weight_format = npu_preparation::get_tensor_npu_format(weight);
        auto grad_weight_format = npu_preparation::get_tensor_npu_format(grad_weight);
        if (grad_weight_format == ACL_FORMAT_NC1HWC0 && weight_format == ACL_FORMAT_ND) {
            npu_format_helper::unsafe_format_cast(grad_weight, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
            npu_format_helper::unsafe_format_cast(grad_bias, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
        }
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace acl_op
