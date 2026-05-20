#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_iou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode)
{
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_iou(bboxes, gtboxes, mode);
    }
    c10::SmallVector<int64_t, SIZE> output_size = {bboxes.size(0), gtboxes.size(0)};
    at::Tensor overlap = npu_preparation::apply_tensor_with_format(output_size, bboxes.options(), ACL_FORMAT_ND);

    const char *mode_str = mode == 1 ? "iof" : "iou";
    float eps = 0.01;
    bool aligned = false;
    EXEC_NPU_CMD(aclnnIou, bboxes, gtboxes, mode_str, eps, aligned, overlap);

    return overlap.transpose(0,1).contiguous();
}
} // namespace op_api
