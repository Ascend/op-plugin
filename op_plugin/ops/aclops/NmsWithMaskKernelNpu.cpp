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

namespace {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> nms_with_mask_npu_nocheck(
    at::Tensor& boxes,
    at::Tensor& idx,
    at::Tensor& mask,
    const at::Tensor& input,
    at::Scalar iou_threshold)
{
    float iou_threshold_value = op_plugin::utils::get_scalar_float_value(iou_threshold);
    at_npu::native::OpCommand cmd;
    cmd.Name("NMSWithMask")
        .Input(input)
        .Output(boxes)
        .Output(idx)
        .Output(mask)
        .Attr("iou_threshold", iou_threshold_value)
        .Run();
    return std::tie(boxes, idx, mask);
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_nms_with_mask(
    const at::Tensor& input,
    const at::Scalar& iou_threshold)
{
    auto output_sizes = op_infer::nms_with_mask_npu_output_size(input);
    at::Tensor boxes = npu_preparation::apply_tensor(input, std::get<0>(output_sizes));
    at::Tensor idx = npu_preparation::apply_tensor(std::get<1>(output_sizes), input.options().dtype(at::kInt), input);
    at::Tensor mask = npu_preparation::apply_tensor(std::get<2>(output_sizes), input.options().dtype(at::kByte), input);
    nms_with_mask_npu_nocheck(boxes, idx, mask, input, iou_threshold);
    return std::tie(boxes, idx, mask);
}

} // namespace acl_op
