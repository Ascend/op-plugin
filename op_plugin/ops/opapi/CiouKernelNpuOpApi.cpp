// Copyright (c) 2026, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, op_infer::SIZE> ciou_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross)
{
    TORCH_CHECK(self.dim() == 2, "ciou expected input in 2D, "
        "but input self has sizes ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gtboxes.dim() == 2, "ciou expected input in 2D, "
        "but input gtboxes has sizes ", gtboxes.dim(),
        OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    if (is_cross) {
        output_size = {gtboxes.size(1), self.size(1)};
    } else {
        output_size = {1, self.size(1)};
    }
    return output_size;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
 	    return acl_op::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
 	}
    bool self_is_half = self.scalar_type() == at::kHalf;
    bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
    at::Tensor self_cp = self_is_half ? at_npu::native::custom_ops::_npu_dtype_cast(self, at::kFloat) : self;
    at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::_npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

    auto output_size = ciou_npu_output_size(self_cp, gtboxes_cp, is_cross);
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);
    at::Tensor atan_sub = npu_preparation::apply_tensor_with_format(output_size, self_cp.options(), ACL_FORMAT_ND);

    const char *mode_str = mode == 1 ? "iof" : "iou";
    EXEC_NPU_CMD(aclnnCIoU, self_cp, gtboxes_cp, trans, is_cross, mode_str, overlap, atan_sub);

    if (self_is_half || gtboxes_is_half) {
        overlap = at_npu::native::custom_ops::_npu_dtype_cast(overlap, at::kHalf);
        atan_sub = at_npu::native::custom_ops::_npu_dtype_cast(atan_sub, at::kHalf);
    }
    return std::tie(overlap, atan_sub);
}

at::Tensor npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
 	    return acl_op::npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
 	}

    auto results = at_npu::native::custom_ops::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
    return std::get<0>(results);
}

std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const c10::optional<at::Tensor>& atan_sub_opt,
    bool trans,
    bool is_cross,
    int64_t mode)
{
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950) {
        TORCH_CHECK(false, "npu_ciou_backward is not supported on Ascend950 and above",
            OPS_ERROR(ErrCode::NOT_SUPPORT));
    }
    return acl_op::npu_ciou_backward(grad, bboxes, gtboxes, atan_sub_opt, trans, is_cross, mode);
}
} // namespace op_api
