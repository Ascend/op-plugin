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
at::SmallVector<int64_t, SIZE> upsample_nearest3d_infer_size(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);

  at::SmallVector<int64_t, SIZE> output_sizes =
      {nbatch, channels, output_depth, output_height, output_width};

  return output_sizes;
}

at::Tensor& upsample_nearest3d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at_npu::native::OpCommand cmd;
  cmd.Name("UpsampleNearest3d")
      .Input(input)
      .Output(result)
      .Attr("output_size", output_size)
      .Run();

  return result;
}
} // namespace

at::Tensor& upsample_nearest3d_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  auto op_infer_output_size = upsample_nearest3d_infer_size(
      input, output_size, scales_d, scales_h, scales_w);
  npu_preparation::CheckOut(
      {input},
      result,
      input,
      op_infer_output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    upsample_nearest3d_out_nocheck(
        contiguous_result, input, output_size, scales_d, scales_h, scales_w);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    upsample_nearest3d_out_nocheck(
        result, input, output_size, scales_d, scales_h, scales_w);
  }
  return result;
}

at::Tensor upsample_nearest3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto op_infer_output_size = upsample_nearest3d_infer_size(
      input, output_size, scales_d, scales_h, scales_w);
  at::Tensor result = npu_preparation::apply_tensor(input, op_infer_output_size);
  upsample_nearest3d_out_nocheck(
      result, input, output_size, scales_d, scales_h, scales_w);
  return result;
}
} // namespace acl_op
