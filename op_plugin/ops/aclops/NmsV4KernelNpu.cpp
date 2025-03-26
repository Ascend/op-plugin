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

std::tuple<at::Tensor&, at::Tensor&> nms_v4_npu_nocheck(
    at::Tensor& selected_indices,
    at::Tensor& valid_outputs,
    const at::Tensor& self,
    const at::Tensor& scores,
    at::Scalar max_output_size,
    const at::Tensor& iou_threshold,
    const at::Tensor& scores_threshold,
    bool pad_to_max_output_size)
{
    at::Tensor max_output_size_tensor = npu_preparation::apply_tensor({}, self.options().dtype(at::kInt), self);
    acl_op::fill_(max_output_size_tensor, max_output_size);
    at_npu::native::OpCommand cmd;
    cmd.Name("NonMaxSuppressionV4")
        .Input(self)
        .Input(scores)
        .Input(max_output_size_tensor)
        .Input(iou_threshold)
        .Input(scores_threshold)
        .Output(selected_indices)
        .Output(valid_outputs)
        .Attr("pad_to_max_output_size", pad_to_max_output_size)
        .Run();
    return std::tie(selected_indices, valid_outputs);
}
}

std::tuple<at::Tensor, at::Tensor> npu_nms_v4(
    const at::Tensor& self,
    const at::Tensor& scores,
    const at::Scalar& max_output_size,
    const at::Tensor& iou_threshold,
    const at::Tensor& scores_threshold,
    bool pad_to_max_output_size)
{
    auto output_sizes = op_infer::nms_v4_npu_output_size(max_output_size);

    at::Tensor selected_indices = npu_preparation::apply_tensor(
        std::get<0>(output_sizes),
        self.options().dtype(at::kInt),
        self);
    at::Tensor valid_outputs = npu_preparation::apply_tensor(
        std::get<1>(output_sizes),
        self.options().dtype(at::kInt),
        self);

    nms_v4_npu_nocheck(
        selected_indices,
        valid_outputs,
        self,
        scores,
        max_output_size,
        iou_threshold,
        scores_threshold,
        pad_to_max_output_size);

    return std::tie(selected_indices, valid_outputs);
}

} // namespace acl_op
