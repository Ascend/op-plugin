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

at::Tensor crop_and_resize(
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    c10::string_view method) {
    TORCH_CHECK(boxes.has_value(), "[boxes] should be mandatory" + OPS_ERROR(ErrCode::VALUE));
    auto output_size = op_infer::crop_and_resize_npu_output_size(self, box_index, crop_size);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);

    std::vector<int64_t> boxes_shape = {boxes->size() / 4, 4};
    at_npu::native::OpCommand cmd;
    cmd.Name("CropAndResizeV2")
        .Input(self)
        .Input(boxes.value(), boxes_shape, at::kFloat)
        .Input(box_index, at::kInt)
        .Input(crop_size, at::kInt)
        .Output(result)
        .Attr<float>("extrapolation_value", extrapolation_value)
        .Attr<std::string>("method", std::string(method).data())
        .Attr("dtype", result.scalar_type())
        .Run();

    return result;
}
} // namespace acl_op
