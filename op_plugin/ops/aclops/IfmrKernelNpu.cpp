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

std::tuple<at::Tensor, at::Tensor> npu_ifmr(
    const at::Tensor& data,
    const at::Tensor& data_min,
    const at::Tensor& data_max,
    const at::Tensor& cumsum,
    const double min_percentile,
    const double max_percentile,
    const double search_start,
    const double search_end,
    const double search_step,
    const bool with_offset) {
  at::Tensor scale = npu_preparation::apply_tensor_with_format(data_min, ACL_FORMAT_NCHW);
  at::Tensor offset = npu_preparation::apply_tensor_with_format(data_min, ACL_FORMAT_NCHW);

  std::vector<float> tmp;
  tmp.push_back(static_cast<float>(search_start));
  tmp.push_back(static_cast<float>(search_end));
  at::ArrayRef<float> search_range(tmp);
  at_npu::native::OpCommand cmd;
  cmd.Name("IFMR")
      .Input(data)
      .Input(data_min)
      .Input(data_max)
      .Input(cumsum)
      .Attr("min_percentile", static_cast<float>(min_percentile))
      .Attr("max_percentile", static_cast<float>(max_percentile))
      .Attr("search_range", search_range)
      .Attr("search_step", static_cast<float>(search_step))
      .Attr("with_offset", with_offset)
      .Output(scale)
      .Output(offset)
      .Run();
  return std::tie(scale, offset);
}
} // namespace acl_op
