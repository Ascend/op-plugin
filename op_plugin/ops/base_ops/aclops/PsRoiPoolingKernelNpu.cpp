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
using npu_op_command = at_npu::native::OpCommand;

namespace {
at::Tensor& ps_roi_pooling_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  npu_op_command cmd;
  cmd.Name("PSROIPoolingV2")
      .Input(self, "x")
      .Input(rois)
      .Output(result, "y")
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("output_dim", output_dim)
      .Attr("group_size", group_size)
      .Run();

  return result;
}
} // namespace


at::Tensor npu_ps_roi_pooling(
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  auto output_size = {rois.size(0) * rois.size(2), output_dim, group_size, group_size};
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  ps_roi_pooling_npu_nocheck(result, self, rois, spatial_scale, group_size, output_dim);
  return result;
}
} // namespace acl_op
