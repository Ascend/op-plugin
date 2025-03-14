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

std::tuple<at::Tensor, at::Tensor> npu_nms_rotated(
    const at::Tensor& self,
    const at::Tensor& scores,
    double iou_threshold,
    double scores_threshold,
    int64_t max_output_size,
    int64_t mode)
{
    // the Op only support fp32 currently!
    auto origin_dtype = self.scalar_type();
    at::Tensor dets_cast = self;
    at::Tensor scores_cast = scores;
    at::Tensor labels = at::zeros({}, scores.options().dtype(at::kInt));
    if (origin_dtype != at::ScalarType::Float) {
        dets_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);
        scores_cast = at_npu::native::custom_ops::npu_dtype_cast(scores, at::kFloat);
    }
    c10::SmallVector<int64_t, SIZE> selected_index_size = {self.size(0)};
    at::Tensor selected_box = npu_preparation::apply_tensor(dets_cast);
    at::Tensor selected_index = npu_preparation::apply_tensor(selected_index_size, self.options().dtype(at::kInt), self);

    c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
    at_npu::native::OpCommand cmd;
    cmd.Sync(output_sync_idx)
        .Name("RotatedNMS")
        .Input(dets_cast)
        .Input(scores_cast)
        .Input(labels)
        .Output(selected_box)
        .Output(selected_index)
        .Attr("iou_threshold", static_cast<float>(iou_threshold))
        .Attr("score_threshold", static_cast<float>(scores_threshold))
        .Attr("max_output_size", max_output_size)
        .Attr("mode", mode)
        .Run();

    at::Tensor selected_num = npu_preparation::apply_tensor({1}, scores.options().dtype(at::kInt), scores);
    acl_op::fill_(selected_num, selected_index.size(0));
    return std::tie(selected_index, selected_num);
}

} // namespace acl_op
