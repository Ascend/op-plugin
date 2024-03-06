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
const int64_t IOF_MODE = 1;

c10::SmallVector<int64_t, N> diou_output_size(const at::Tensor &self, const at::Tensor &gtboxes, bool is_cross)
{
    c10::SmallVector<int64_t, N> output_size;
    TORCH_CHECK(self.dim() == 2, "self has to be a 2D Tensor, but got Tensor of dimension ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gtboxes.dim() == 2, "gtboxes has to be a 2D Tensor, but got Tensor of dimension ", gtboxes.dim(),
        OPS_ERROR(ErrCode::PARAM));
    if (is_cross) {
        output_size = {gtboxes.size(1), self.size(1)};
    } else {
        output_size = {1, self.size(1)};
    }
    return output_size;
}

at::Tensor &diou_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &gtboxes, bool trans,
                                 bool is_cross, int64_t mode)
{
    auto output_size = diou_output_size(self, gtboxes, is_cross);
    string mode_str = mode == IOF_MODE ? "iof" : "iou";

    at_npu::native::OpCommand cmd;
    cmd.Name("DIoU")
        .Input(self)
        .Input(gtboxes)
        .Output(result)
        .Attr("trans", trans)
        .Attr("is_cross", is_cross)
        .Attr("mode", mode_str)
        .Run();
    return result;
}

at::Tensor diou_npu_nocheck(const at::Tensor &self, const at::Tensor &gtboxes, bool trans, bool is_cross, int64_t mode)
{
    // Op need form of [n, 4], but pass should be [4, n];
    // Note: temp avoid! it'll be removed while op deal with fp16 issue!
    bool self_is_half = self.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor self_cp = self_is_half ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat) : self;
    at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

    auto output_size = diou_output_size(self_cp, gtboxes_cp, is_cross);
    at::Tensor result = npu_preparation::apply_tensor(self_cp, output_size);
    diou_out_npu_nocheck(result, self_cp, gtboxes_cp, trans, is_cross, mode);

    if (self_is_half || gtboxes_is_half) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kHalf);
    }
    return result;
}

std::tuple<at::Tensor &, at::Tensor &> diou_backward_out_npu_nocheck(at::Tensor &dbboxes, at::Tensor &dgtboxes,
                                                                     const at::Tensor &grad, const at::Tensor &bboxes,
                                                                     const at::Tensor &gtboxes, bool trans,
                                                                     bool is_cross, int64_t mode)
{
    string mode_str = mode == IOF_MODE ? "iof" : "iou";
    at_npu::native::OpCommand cmd;
    cmd.Name("DIoUGrad")
        .Input(grad)
        .Input(bboxes)
        .Input(gtboxes)
        .Output(dbboxes)
        .Output(dgtboxes)
        .Attr("trans", trans)
        .Attr("is_cross", is_cross)
        .Attr("mode", mode_str)
        .Run();
    return std::tie(dbboxes, dgtboxes);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_diou_backward(const at::Tensor &grad, const at::Tensor &bboxes,
                                                     const at::Tensor &gtboxes, bool trans, bool is_cross, int64_t mode)
{
    // Op need form of [n] grad
    // Note: temp avoid! it'll be remove while op deal with fp16 issue!
    at::Tensor grad_cp = at::squeeze(grad, 0);
    if (grad_cp.scalar_type() == at::kHalf) {
        grad_cp = at_npu::native::custom_ops::npu_dtype_cast(grad_cp, at::kFloat);
    }

    bool bboxes_is_half = bboxes.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor bboxes_cp = bboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(bboxes, at::kFloat) : bboxes;
    at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;
    at::Tensor dbboxes = npu_preparation::apply_tensor(bboxes_cp);
    at::Tensor dgtboxes = npu_preparation::apply_tensor(gtboxes_cp);

    diou_backward_out_npu_nocheck(dbboxes, dgtboxes, grad_cp, bboxes_cp, gtboxes_cp, trans, is_cross, mode);
    if (bboxes_is_half || gtboxes_is_half) {
        dbboxes = at_npu::native::custom_ops::npu_dtype_cast(dbboxes, at::kHalf);
        dgtboxes = at_npu::native::custom_ops::npu_dtype_cast(dgtboxes, at::kHalf);
    }
    return std::tie(dbboxes, dgtboxes);
}

at::Tensor npu_diou(const at::Tensor &self, const at::Tensor &gtboxes, bool trans, bool is_cross, int64_t mode)
{
    return diou_npu_nocheck(self, gtboxes, trans, is_cross, mode);
}
} // namespace acl_op
