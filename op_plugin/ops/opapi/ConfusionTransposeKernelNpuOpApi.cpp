// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline c10::SmallVector<int64_t, SIZE> transpose_shape(
    at::IntArrayRef ori_shape, at::IntArrayRef perm, bool transpose_first)
{
    c10::SmallVector<int64_t, SIZE> trans_shape;
    if (transpose_first) {
        trans_shape = op_infer::array_to_small_vector(ori_shape);
    } else {
        size_t shape_size = ori_shape.size();
        for (size_t i = 0; i < perm.size(); i++) {
            TORCH_CHECK(shape_size > perm[i],
                "npu_confusion_transpose forward/backward input invalid, "
                "shape has size ",
                shape_size,
                " but perm[i] is, ",
                perm[i],
                OPS_ERROR(ErrCode::PARAM));
            trans_shape.emplace_back(ori_shape[perm[i]]);
        }
    }
    return trans_shape;
}

at::Tensor npu_confusion_transpose(
    const at::Tensor &self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_confusion_transpose(self, perm, shape, transpose_first);
    }
    DO_COMPATIBILITY(aclnnConfusionTranspose, acl_op::npu_confusion_transpose(self, perm, shape, transpose_first));

    c10::SmallVector<int64_t, SIZE> svec_output_shape = transpose_shape(shape, perm, transpose_first);
    at::Tensor y = npu_preparation::apply_tensor_without_format(svec_output_shape, self.options());
    EXEC_NPU_CMD(aclnnConfusionTranspose, self, perm, shape, transpose_first, y);
    return y;
}

at::Tensor npu_confusion_transpose_backward_symint(
    const at::Tensor &grad, at::IntArrayRef perm, c10::SymIntArrayRef shape_symint, bool transpose_first)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_confusion_transpose_backward_symint(grad, perm, shape_symint, transpose_first);
    }
    DO_COMPATIBILITY(aclnnConfusionTranspose,
        acl_op::npu_confusion_transpose_backward_symint(grad, perm, shape_symint, transpose_first));

    at::IntArrayRef shape = c10::asIntArrayRefUnchecked(shape_symint);
    c10::SmallVector<int64_t, SIZE> svec_backward_shape = transpose_shape(shape, perm, transpose_first);
    at::IntArrayRef backward_shape(svec_backward_shape);

    int64_t perm_len = perm.size();
    c10::SmallVector<int64_t, SIZE> svec_backward_perm(perm_len, 0);
    for (int64_t i = 0; i < perm_len; i++) {
        svec_backward_perm[perm[i]] = i;
    }
    at::IntArrayRef backward_perm(svec_backward_perm);

    at::Tensor y = npu_preparation::apply_tensor_without_format(shape, grad.options());
    EXEC_NPU_CMD(aclnnConfusionTranspose, grad, backward_perm, backward_shape, transpose_first, y);
    return y;
}
}  // namespace op_api
