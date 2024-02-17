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

at::Tensor npu_bounding_box_encode(
    const at::Tensor& anchor_box,
    const at::Tensor& ground_truth_box,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3) {
  at::Tensor result = npu_preparation::apply_tensor(anchor_box, {anchor_box.size(0), 4});
  c10::SmallVector<float, SIZE> means = {
      static_cast<float>(means0),
      static_cast<float>(means1),
      static_cast<float>(means2),
      static_cast<float>(means3)};
  c10::SmallVector<float, SIZE> stds = {
      static_cast<float>(stds0),
      static_cast<float>(stds1),
      static_cast<float>(stds2),
      static_cast<float>(stds3)};
  at_npu::native::OpCommand cmd;
  cmd.Name("BoundingBoxEncode")
      .Input(anchor_box)
      .Input(ground_truth_box)
      .Output(result)
      .Attr("means", means)
      .Attr("stds", stds)
      .Run();
  return result;
}
} // namespace acl_op
