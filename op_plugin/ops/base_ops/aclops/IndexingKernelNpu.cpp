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
at::Tensor &npu_indexing_out_nocheck(at::Tensor &result, const at::Tensor &self, c10::IntArrayRef begin,
                                     c10::IntArrayRef end, c10::IntArrayRef strides, int64_t begin_mask,
                                     int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                                     int64_t shrink_axis_mask)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("StridedSlice")
        .Input(self)
        .Input(begin)
        .Input(end)
        .Input(strides)
        .Output(result)
        .Attr("begin_mask", begin_mask)
        .Attr("end_mask", end_mask)
        .Attr("ellipsis_mask", ellipsis_mask)
        .Attr("new_axis_mask", new_axis_mask)
        .Attr("shrink_axis_mask", shrink_axis_mask)
        .Run();
    return result;
}

c10::SmallVector<int64_t, SIZE> infersize_npu_indexing(const at::Tensor &self, c10::IntArrayRef begin,
                                                       c10::IntArrayRef end, c10::IntArrayRef strides)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    for (int i = 0; i < self.dim(); i++) {
        TORCH_CHECK(strides[i] != 0, "stride should not be 0"
            + OPS_ERROR(ErrCode::VALUE));
        output_size.emplace_back((end[i] + strides[i] - 1 - begin[i]) / strides[i]);
    }
    return output_size;
}
} // namespace

at::Tensor &npu_indexing_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                             c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                             int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor &result)
{
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        npu_indexing_out_nocheck(contiguous_result, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
                                 new_axis_mask, shrink_axis_mask);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        npu_indexing_out_nocheck(result, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                 shrink_axis_mask);
    }
    return result;
}

at::Tensor npu_indexing(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end, c10::IntArrayRef strides,
                        int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                        int64_t shrink_axis_mask)
{
    auto output_size = infersize_npu_indexing(self, begin, end, strides);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    npu_indexing_out_nocheck(result, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                             shrink_axis_mask);
    return result;
}

at::Tensor &npu_indexing_trans_contiguous_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                                              c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask,
                                              int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask,
                                              at::Tensor &result)
{
    npu_indexing_out_nocheck(result, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                             shrink_axis_mask);
    return result;
}

} // namespace acl_op
