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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::tuple<at::Tensor, at::Tensor> deformable_conv2d_nocheck(const at::Tensor &input, const at::Tensor &weight,
                                                             const at::Tensor &offset,
                                                             const c10::optional<at::Tensor> &bias_opt,
                                                             at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                             at::IntArrayRef padding, at::IntArrayRef dilation,
                                                             int64_t groups, int64_t deformable_groups, bool modulated)
{
    TORCH_CHECK(input.dim() >= 4, "input has to be more than 4D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(offset.dim() >= 4, "offset has to more than 4D, but got Tensor of dimension ", offset.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 4, "stride has to contain more than 4 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 4, "dilation has to contain more than 4 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    at::Tensor bias_fp32 = (bias.defined() && bias.dtype() != at::kFloat) ?
                               at_npu::native::custom_ops::npu_dtype_cast(bias, at::kFloat) :
                               bias;
    auto output_size = op_infer::deformable_conv2d_npu_output_size(input, offset, kernel_size);
    TORCH_CHECK(output_size.size() >= 4,
        "output_size has to contain more than 4 elements, but got Tensor of dimension ", output_size.size(),
        OPS_ERROR(ErrCode::PARAM));
    /*
     * DeformableOffsets and DeformableOffsetsGrad only support NHWC and don't support binary.
     * FE will insert Transpose before DeformableOffsets and DeformableOffsetsGrad.
     * In order to allow Transpose into binary,
     * Transpose is called explicitly in adapter.
     */
    c10::SmallVector<int64_t, SIZE> nhwc_deformable_offsets_output_shape = {output_size[0], output_size[2],
                                                                            output_size[3], output_size[1]};
    at::Tensor nhwc_deformable_offsets_output = npu_preparation::apply_tensor_with_format(
        nhwc_deformable_offsets_output_shape, input.options(), ACL_FORMAT_NHWC);
    c10::SmallVector<int64_t, SIZE> in_perm = {0, 2, 3, 1};
    at::Tensor ori_input = npu_preparation::CastBackToOriFormat(input);
    at::Tensor ori_offset = npu_preparation::CastBackToOriFormat(offset);
    at::Tensor nhwc_input = acl_op::npu_transpose(ori_input, in_perm, true);
    at::Tensor nhwc_offset = acl_op::npu_transpose(ori_offset, in_perm, true);

    auto &nhwc_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_input)->npu_desc_;
    auto &nhwc_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_offset)->npu_desc_;

    nhwc_input_desc.npu_format_ = ACL_FORMAT_NHWC;
    nhwc_input_desc.origin_format_ = ACL_FORMAT_NHWC;
    nhwc_offset_desc.npu_format_ = ACL_FORMAT_NHWC;
    nhwc_offset_desc.origin_format_ = ACL_FORMAT_NHWC;

    c10::SmallVector<int64_t, SIZE> nhwc_strides = {stride[0], stride[2], stride[3], stride[1]};
    c10::SmallVector<int64_t, SIZE> nhwc_dilations = {dilation[0], dilation[2], dilation[3], dilation[1]};
    string data_format = "NHWC";
    at_npu::native::OpCommand cmd;
    cmd.Name("DeformableOffsets")
        .Input(nhwc_input, "X")
        .Input(nhwc_offset, "offsets")
        .Output(nhwc_deformable_offsets_output, "y")
        .Attr("ksize", kernel_size)
        .Attr("strides", nhwc_strides)
        .Attr("pads", padding)
        .Attr("dilations", nhwc_dilations)
        .Attr("deformable_groups", deformable_groups)
        .Attr("data_format", data_format)
        .Attr("modulated", modulated)
        .Run();

    c10::SmallVector<int64_t, SIZE> out_perm = {0, 3, 1, 2};
    nhwc_input_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_input_desc.origin_format_ = ACL_FORMAT_NCHW;
    nhwc_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
    auto &nhwc_deformable_offsets_output_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_deformable_offsets_output)->npu_desc_;
    nhwc_deformable_offsets_output_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_deformable_offsets_output_desc.origin_format_ = ACL_FORMAT_NCHW;
    at::Tensor deformable_offsets_output = acl_op::npu_transpose(nhwc_deformable_offsets_output, out_perm, true);

    c10::SmallVector<int64_t, SIZE> conv2d_stride = op_infer::array_to_small_vector(kernel_size);
    c10::SmallVector<int64_t, SIZE> conv2d_padding = {0, 0, 0, 0};
    c10::SmallVector<int64_t, SIZE> conv2d_dilation = {1, 1};
    at::Tensor conv2d_output = acl_op::npu_conv2d(deformable_offsets_output, weight, bias_fp32, conv2d_stride,
                                                  conv2d_padding, conv2d_dilation, groups);

    return std::tie(conv2d_output, deformable_offsets_output);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deformable_conv2dbk(
    const at::Tensor &input_ori, const at::Tensor &grad_output_ori, const at::Tensor &offset_out_ori,
    const at::Tensor &weight_ori, const at::Tensor &offset_ori, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated)
{
    TORCH_CHECK(input_ori.dim() >= 4, "input_ori has to be more than 4D, but got Tensor of dimension ",
        input_ori.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(offset_ori.dim() >= 4, "offset_ori has to more than 4D, but got Tensor of dimension ",
        offset_ori.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(stride.size() >= 4, "stride has to contain more than 4 elements, but got ", stride.size(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilation.size() >= 4, "dilation has to contain more than 4 elements, but got ", dilation.size(),
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor input = (input_ori.dtype() != at::kFloat) ?
                           at_npu::native::custom_ops::npu_dtype_cast(input_ori, at::kFloat) :
                           input_ori;
    at::Tensor grad_output = (grad_output_ori.dtype() != at::kFloat) ?
                                 at_npu::native::custom_ops::npu_dtype_cast(grad_output_ori, at::kFloat) :
                                 grad_output_ori;
    at::Tensor offset_out = (offset_out_ori.dtype() != at::kFloat) ?
                                at_npu::native::custom_ops::npu_dtype_cast(offset_out_ori, at::kFloat) :
                                offset_out_ori;
    at::Tensor weight = (weight_ori.dtype() != at::kFloat) ?
                            at_npu::native::custom_ops::npu_dtype_cast(weight_ori, at::kFloat) :
                            weight_ori;
    at::Tensor offset = (offset_ori.dtype() != at::kFloat) ?
                            at_npu::native::custom_ops::npu_dtype_cast(offset_ori, at::kFloat) :
                            offset_ori;
    // deformable_conv2d_backward includes conv2d_backward and DeformableOffsetsGrad
    c10::SmallVector<int64_t, SIZE> conv2d_stride = op_infer::array_to_small_vector(kernel_size);
    c10::SmallVector<int64_t, SIZE> conv2d_padding = {0, 0, 0, 0};
    c10::SmallVector<int64_t, SIZE> conv2d_dilation = {1, 1};
    auto conv2d_backward_output = acl_op::npu_conv2d_backward(
        offset_out, grad_output, weight, conv2d_stride, conv2d_padding, conv2d_dilation, groups, {true, true, true});

    // DeformableOffsetsGrad's input 'grad' is the output[0] of conv2d_backward
    at::Tensor deformable_offsets_backward_input = std::get<0>(conv2d_backward_output);
    at::Tensor grad_weight = std::get<1>(conv2d_backward_output);
    at::Tensor grad_bias = std::get<2>(conv2d_backward_output);

    c10::SmallVector<int64_t, SIZE> in_perm = {0, 2, 3, 1};
    auto nhwc_grad_input_shape = op_infer::transpose_npu_output_size(input, in_perm);
    auto nhwc_grad_offset_shape = op_infer::transpose_npu_output_size(offset, in_perm);
    at::Tensor nhwc_grad_input =
        npu_preparation::apply_tensor_with_format(input, nhwc_grad_input_shape, ACL_FORMAT_NHWC);
    at::Tensor nhwc_grad_offset =
        npu_preparation::apply_tensor_with_format(offset, nhwc_grad_offset_shape, ACL_FORMAT_NHWC);

    auto trans_shape = op_infer::transpose_npu_output_size(deformable_offsets_backward_input, in_perm);

    at::Tensor ori_deformable_offsets_backward_input =
        npu_preparation::CastBackToOriFormat(deformable_offsets_backward_input);
    at::Tensor ori_input = npu_preparation::CastBackToOriFormat(input);
    at::Tensor ori_offset = npu_preparation::CastBackToOriFormat(offset);

    at::Tensor nhwc_deformable_offsets_backward_input =
        acl_op::npu_transpose(ori_deformable_offsets_backward_input, in_perm, true);
    at::Tensor nhwc_input = acl_op::npu_transpose(ori_input, in_perm, true);
    at::Tensor nhwc_offset = acl_op::npu_transpose(ori_offset, in_perm, true);

    auto &nhwc_deformable_offsets_backward_input_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_deformable_offsets_backward_input)->npu_desc_;
    auto &nhwc_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_input)->npu_desc_;
    auto &nhwc_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_offset)->npu_desc_;

    nhwc_deformable_offsets_backward_input_desc.npu_format_ = ACL_FORMAT_NHWC;
    nhwc_deformable_offsets_backward_input_desc.origin_format_ = ACL_FORMAT_NHWC;
    nhwc_input_desc.npu_format_ = ACL_FORMAT_NHWC;
    nhwc_input_desc.origin_format_ = ACL_FORMAT_NHWC;
    nhwc_offset_desc.npu_format_ = ACL_FORMAT_NHWC;
    nhwc_offset_desc.origin_format_ = ACL_FORMAT_NHWC;

    c10::SmallVector<int64_t, SIZE> nhwc_strides = {stride[0], stride[2], stride[3], stride[1]};
    c10::SmallVector<int64_t, SIZE> nhwc_dilations = {dilation[0], dilation[2], dilation[3], dilation[1]};
    string data_format = "NHWC";
    at_npu::native::OpCommand cmd;
    cmd.Name("DeformableOffsetsGrad")
        .Input(nhwc_deformable_offsets_backward_input, "grad")
        .Input(nhwc_input, "X")
        .Input(nhwc_offset, "offsets")
        .Output(nhwc_grad_input, "grad_X")
        .Output(nhwc_grad_offset, "grad_offsets")
        .Attr("strides", nhwc_strides)
        .Attr("pads", padding)
        .Attr("ksize", kernel_size)
        .Attr("dilations", nhwc_dilations)
        .Attr("data_format", data_format)
        .Attr("deformable_groups", deformable_groups)
        .Attr("modulated", modulated)
        .Run();
    c10::SmallVector<int64_t, SIZE> out_perm = {0, 3, 1, 2};
    nhwc_deformable_offsets_backward_input_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_deformable_offsets_backward_input_desc.origin_format_ = ACL_FORMAT_NCHW;
    nhwc_input_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_input_desc.origin_format_ = ACL_FORMAT_NCHW;
    nhwc_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
    auto &nhwc_grad_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_grad_input)->npu_desc_;
    auto &nhwc_grad_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_grad_offset)->npu_desc_;
    nhwc_grad_input_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_grad_input_desc.origin_format_ = ACL_FORMAT_NCHW;
    nhwc_grad_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
    nhwc_grad_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
    at::Tensor grad_input = acl_op::npu_transpose(nhwc_grad_input, out_perm, true);
    at::Tensor grad_offset = acl_op::npu_transpose(nhwc_grad_offset, out_perm, true);

    return std::tie(grad_input, grad_weight, grad_offset, grad_bias);
}

std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d(const at::Tensor &input_ori, const at::Tensor &weight_ori,
                                                         const at::Tensor &offset_ori,
                                                         const c10::optional<at::Tensor> &bias_opt,
                                                         at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                         at::IntArrayRef padding, at::IntArrayRef dilation,
                                                         int64_t groups, int64_t deformable_groups, bool modulated)
{
    at::Tensor input = (input_ori.dtype() != at::kFloat) ?
                           at_npu::native::custom_ops::npu_dtype_cast(input_ori, at::kFloat) :
                           input_ori;
    at::Tensor weight = (weight_ori.dtype() != at::kFloat) ?
                            at_npu::native::custom_ops::npu_dtype_cast(weight_ori, at::kFloat) :
                            weight_ori;
    at::Tensor offset = (offset_ori.dtype() != at::kFloat) ?
                            at_npu::native::custom_ops::npu_dtype_cast(offset_ori, at::kFloat) :
                            offset_ori;
    return deformable_conv2d_nocheck(input, weight, offset, bias_opt, kernel_size, stride, padding, dilation, groups,
                                     deformable_groups, modulated);
}
} // namespace acl_op
