// Copyright (c) 2023 Huawei Technologies Co., Ltd
// 版权所有 (c) 2023 华为技术有限公司
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
at::Tensor& batch_norm_infer_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("BNInfer")
        .Input(self, "x")
        .Input(weight, "scale")
        .Input(bias, "offset")
        .Input(running_mean, "mean")
        .Input(running_var, "variance")
        .Output(result, "y")
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    return result;
}

std::tuple<at::Tensor&, at::Tensor&> batch_norm_training_reduce_nocheck(
    at::Tensor& sum,
    at::Tensor& square_sum,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps)
{
    at_npu::native::OpCommand cmd;
    string name = (self.dim() == 5) ? "BN3DTrainingReduce" : "BNTrainingReduce";
    cmd.Name(name)
        .Input(self, "x")
        .Output(sum, "sum")
        .Output(square_sum, "square_sum")
        .Attr("epsilon", static_cast<float>(eps))
        .Run();

    return std::tie(sum, square_sum);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_training_update_nocheck(
    at::Tensor& result,
    at::Tensor& save_mean,
    at::Tensor& save_invstd,
    const at::Tensor& self,
    const at::Tensor& sum,
    const at::Tensor& square_sum,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps)
{
    at_npu::native::OpCommand cmd;
    string name = (self.dim() == 5) ? "BN3DTrainingUpdate" : "BNTrainingUpdate";
    cmd.Name(name)
        .Input(self, "x")
        .Input(sum, "sum")
        .Input(square_sum, "square_sum")
        .Input(weight, "scale")
        .Input(bias, "offset")
        .Input(running_mean, "mean")
        .Input(running_var, "variance")
        .Output(result, "y")
        .Output(const_cast<at::Tensor&>(running_mean), "mean")
        .Output(const_cast<at::Tensor&>(running_var), "variance")
        .Output(save_mean, "batch_mean")
        .Output(save_invstd, "batch_variance")
        .Attr("epsilon", static_cast<float>(eps))
        .Attr("factor", static_cast<float>(momentum))
        .Run();

    return std::tie(result, save_mean, save_invstd);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_impl(
    at::Tensor& result,
    at::Tensor& save_mean,
    at::Tensor& save_invstd,
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps)
{
    if (!train) {
        batch_norm_infer_nocheck(result, self, weight, bias, running_mean, running_var, train, momentum, eps);
        return std::tie(result, save_mean, save_invstd);
    }

    at::Tensor sum = (self.dim() == 5) ?
        npu_preparation::apply_tensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean) :
        npu_preparation::apply_tensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), self);
    at::Tensor square_sum = (self.dim() == 5) ?
        npu_preparation::apply_tensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), running_mean) :
        npu_preparation::apply_tensor(running_mean.sizes(), running_mean.options().dtype(at::kFloat), self);

    batch_norm_training_reduce_nocheck(
        sum, square_sum, self, weight, bias, running_mean, running_var, train, momentum, eps);

    // BNTrainingUpdate can only support FP32 for mean and var
    auto running_mean_fp32 = running_mean;
    auto running_var_fp32 = running_var;
    auto weight_fp32 = weight;

    if (train && (running_mean.scalar_type() != at::kFloat)) {
        running_mean_fp32 = at_npu::native::custom_ops::npu_dtype_cast(running_mean, at::kFloat);
    }

    if (train && (running_var.scalar_type() != at::kFloat)) {
        running_var_fp32 = at_npu::native::custom_ops::npu_dtype_cast(running_var, at::kFloat);
    }

    // (Ascend) change dtype for matching op requirement.
    if (train && (weight.scalar_type() != at::kFloat)) {
        weight_fp32 = at_npu::native::custom_ops::npu_dtype_cast(weight, at::kFloat);
    }
    at::Tensor bias_cp = bias;
    auto self_format = npu_preparation::get_tensor_npu_format(self);
    auto weight_format = npu_preparation::get_tensor_npu_format(weight_fp32);

    bool check_bn_5hd = (self_format == ACL_FORMAT_NC1HWC0 && weight_format == ACL_FORMAT_ND) ? true : false;
    if (check_bn_5hd) {
        npu_format_helper::unsafe_format_cast(weight_fp32, ACL_FORMAT_ND, ACL_FORMAT_NC1HWC0);
        npu_format_helper::unsafe_format_cast(bias_cp, ACL_FORMAT_ND, ACL_FORMAT_NC1HWC0);
        npu_format_helper::unsafe_format_cast(running_mean_fp32, ACL_FORMAT_ND, ACL_FORMAT_NC1HWC0);
        npu_format_helper::unsafe_format_cast(running_var_fp32, ACL_FORMAT_ND, ACL_FORMAT_NC1HWC0);
    }

    batch_norm_training_update_nocheck(
        result, save_mean, save_invstd, self, sum, square_sum, weight_fp32, bias_cp, running_mean_fp32, running_var_fp32,
        train, momentum, eps);

    if (check_bn_5hd) {
        npu_format_helper::unsafe_format_cast(weight_fp32, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
        npu_format_helper::unsafe_format_cast(bias_cp, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
        npu_format_helper::unsafe_format_cast(running_mean_fp32, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
        npu_format_helper::unsafe_format_cast(running_var_fp32, ACL_FORMAT_NC1HWC0, ACL_FORMAT_ND);
    }

    return std::tie(result, save_mean, save_invstd);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps)
{
    int64_t dim_c = self.size(1);
    at::TensorOptions options = self.options().dtype(at::kFloat);
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});
    const at::Tensor running_mean_tensor = running_mean.defined() ? running_mean : at::zeros({dim_c}, options);
    const at::Tensor running_var_tensor = running_var.defined() ? running_var : at::ones({dim_c}, options);

    at::Tensor result;
    at::Tensor save_mean;
    at::Tensor save_invstd;
    if (train) {
        save_mean = (self.dim() == 5) ?
            npu_preparation::apply_tensor(
                running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat), running_mean_tensor) :
            npu_preparation::apply_tensor(
                running_mean_tensor.sizes(), running_mean_tensor.options().dtype(at::kFloat), self);
        save_invstd = (self.dim() == 5) ?
            npu_preparation::apply_tensor(
                running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat), running_var_tensor) :
            npu_preparation::apply_tensor(
                running_var_tensor.sizes(), running_var_tensor.options().dtype(at::kFloat), self);
    } else {
        save_mean = at::empty({0}, self.options());
        save_invstd = at::empty({0}, self.options());
    }

    return acl_op::native_batch_norm_out(self, weight_opt, bias_opt,
        running_mean_opt, running_var_opt, train, momentum, eps, result, save_mean, save_invstd);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps,
    at::Tensor& result,
    at::Tensor& save_mean,
    at::Tensor& save_invstd)
{
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] { return at::Tensor(); });
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

    at::Tensor self_reshape;
    c10::SmallVector<int64_t, N> self_shape = op_infer::array_to_small_vector(self.sizes());

    int64_t self_npu_format = npu_preparation::get_tensor_npu_format(self);
    // BatchNorm is axis sensitive, the size of mean/var depends on dim_c.
    TORCH_CHECK(
        !(self_npu_format == ACL_FORMAT_NDHWC || self_npu_format == ACL_FORMAT_NHWC),
        "at::Tensor with channel last format (",
        self_npu_format,
        ") is not supported in BatchNorm." + OPS_ERROR(ErrCode::TYPE));

    if (self.dim() <= 4) {
        c10::SmallVector<int64_t, N> nchw_shape(self_shape);
        nchw_shape.resize(4, 1);
        self_reshape = self.reshape(nchw_shape);
        if (result.defined()) {
            result = result.reshape(nchw_shape);
        }
    } else if (train && self.dim() == 5) {
        // Use 3D BN ops for training, merging axes is not required.
        self_reshape = self;
    } else {
        // Infering uses 2dInfer Op, case no matched 3DInfer Op
        // ncdhw -> ndchw
        self_reshape = self.permute({0, 2, 1, 3, 4});
        // nchw=(n*d, c, h, w)
        c10::SmallVector<int64_t, N> nchw_shape =
            {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
        // ndchw -> nchw
        self_reshape = self_reshape.reshape(nchw_shape);
        if (result.defined()) {
            result = npu_preparation::apply_tensor(self_reshape);
        }
    }

    // process when affine=Flase and track_running_stats=False
    int64_t dim_c = self_reshape.size(1);
    at::TensorOptions options = self.options().dtype(at::ScalarType::Float);

    at::Tensor weight_cp = weight;
    at::Tensor bias_cp = bias;
    at::Tensor running_mean_cp = running_mean;
    at::Tensor running_var_cp = running_var;

    // 2D/3D BN Ops support ACL_FORMAT_NC1HWC0 format tensor(1D).
    at::Tensor running_mean_tensor = running_mean.defined() ? running_mean_cp : at::zeros({dim_c}, options);
    at::Tensor running_var_tensor = running_var.defined() ? running_var_cp : at::ones({dim_c}, options);
    at::Tensor weight_tensor = weight.defined() ? weight_cp : at::ones({dim_c}, options);
    at::Tensor bias_tensor = bias.defined() ? bias_cp : at::zeros({dim_c}, options);

    if (!result.defined()) {
        result = npu_preparation::apply_tensor(self_reshape);
    }

    batch_norm_impl(result, save_mean, save_invstd, self_reshape, weight_tensor, bias_tensor, running_mean_tensor,
        running_var_tensor, train, momentum, eps);

    // Inverse reshape procedure using for recovering original shape of self.
    if (!train && self.dim() == 5) {
        // NCHW -> NDCHW -> NCDHW
        std::swap(self_shape[1], self_shape[2]);
        result = result.view(self_shape);
        result = npu_utils::format_contiguous(result);
        result = result.permute({0, 2, 1, 3, 4}).clone();
    } else if (self.dim() < 5) {
        result = result.view(self_shape);
        result = npu_utils::format_contiguous(result);
    }

    return std::tie(result, save_mean, save_invstd);
}

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps)
{
    return acl_op::native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps);
}
#endif
} // namespace acl_op
