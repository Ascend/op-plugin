// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include <ATen/native/ForeachUtils.h>

#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "op_plugin/utils/custom_functions/opapi/ForeachConstants.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const size_t SIZE_OF_NOT_INT = 4;
const size_t SIZE_OF_SHORT = 2;
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;


void exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    EXEC_NPU_CMD(aclnnForeachCopy, src, dst);
}

void split_and_exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    size_t tensor_count = src.size();
    size_t max_tensor_count = SINGLE_FOREACH_OP_TENSOR_COUNT;
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        exec_npu_cmd_copy(dst, src, non_blocking);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_src(src.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_dst(dst.data() + i * max_tensor_count, max_tensor_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count != 0) {
        at::TensorList temp_src(src.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_dst(dst.data() + loop_time * max_tensor_count, remaining_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }
}

bool check_tensor_dtype_support_base(const at::TensorList src)
{
    if ((sizeof(src[0]) == SIZE_OF_NOT_INT && src[0].scalar_type() != at::ScalarType::QInt32) ||
         src[0].scalar_type() == at::ScalarType::Int) {
        return true;
    }
    if (sizeof(src[0]) == SIZE_OF_SHORT || src[0].scalar_type() == at::ScalarType::Short) {
        return true;
    }
    if (src[0].scalar_type() == at::ScalarType::Char || src[0].scalar_type() == at::ScalarType::Byte ||
        src[0].scalar_type() == at::ScalarType::BFloat16 ||
        src[0].scalar_type() == at::ScalarType::Float || src[0].scalar_type() == at::ScalarType::Half) {
        return true;
    } else if (op_plugin::utils::is_gte_cann_version_810rc1() && (src[0].scalar_type() == at::ScalarType::Long ||
            src[0].scalar_type() == at::ScalarType::Double || src[0].scalar_type() == at::ScalarType::Bool)) {
        return true;
    }
    return false;
}

bool check_tensor_device_dtype_base(const at::TensorList dsts, const at::TensorList srcs)
{
    if (dsts.size() != srcs.size() || dsts.size() == 0 || srcs.size() == 0) {
        return false;
    }
    // 方向一致校验
    const auto expected_dst_dtype =  dsts[0].device().type();
    const auto expected_src_dtype =  srcs[0].device().type();
    for (const auto &dst: dsts) {
        if (dst.device().type() != expected_dst_dtype) {
            return false;
        }
    }
    for (const auto &src: srcs) {
        if (src.device().type() != expected_src_dtype) {
            return false;
        }
    }
    // 排除d2d场景
    if (expected_dst_dtype == expected_src_dtype) {
        return false;
    }
    return true;
}

void memcpyBatch(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    TORCH_CHECK(dst.size() == src.size(), "dst and src size,must be equal but in realiry, the dst size is", dst.size(),
                " and the src size is ", dst.size(), "." + OPS_ERROR(ErrCode::PARAM));
    size_t count = dst.size();
    void *dsts[count];
    void *srcs[count];
    size_t dstLens[count];
    size_t srcLens[count];
    size_t attrsIndexes[count];
    aclrtMemcpyBatchAttr attrs[count];
    for (size_t i = 0; i < count; ++i) {
        aclrtMemcpyBatchAttr attr;
        aclrtMemLocation dstLoc;
        aclrtMemLocation srcLoc;
        at::Tensor dst_tensor = dst[i];
        at::Tensor src_tensor = src[i];
        // 获取 Tensor 的地址
        dsts[i] = dst_tensor.data_ptr();
        srcs[i] = src_tensor.data_ptr();
        // 计算 Tensor 的内存大小
        dstLens[i] = static_cast<size_t>(dst_tensor.numel() * dst_tensor.element_size());
        srcLens[i] = static_cast<size_t>(src_tensor.numel() * src_tensor.element_size());
        attrsIndexes[i] = i;
        // 判断哪个是d哪个是h
        if (dst_tensor.device().type() == c10::DeviceType::PrivateUse1) {
            int npu_device_index = dst_tensor.device().index();
            dstLoc.id = npu_device_index;
            dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
            attr.dstLoc = dstLoc;
            srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
            attr.srcLoc = srcLoc;
        };
        if (src_tensor.device().type() == c10::DeviceType::PrivateUse1) {
            int npu_device_index = src_tensor.device().index();
            srcLoc.id = npu_device_index;
            srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
            attr.srcLoc = srcLoc;
            dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
            attr.dstLoc = dstLoc;
        }
        constexpr uint32_t rsvMaxSize = sizeof(aclrtMemcpyBatchAttr::rsv) / sizeof(uint8_t);
        for (uint32_t j = 0U; j < rsvMaxSize; j++) {
            attr.rsv[j] = 0U;
        }
        attrs[i] = attr;
    }
    size_t failIdx = SIZE_MAX;
    auto acl_stream = c10_npu::getCurrentNPUStream().stream();
    if (non_blocking) {
        auto ret = c10_npu::acl::AclrtMemcpyBatchAsync(dsts, dstLens, srcs, srcLens, count, attrs, attrsIndexes, count,
                                                       &failIdx, acl_stream);
        NPU_CHECK_ERROR(ret, "aclrtMemcpyBatchAsync");
    } else {
        aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(acl_stream);
        if (error != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
            C10_NPU_SHOW_ERR_MSG();
            if (c10_npu::option::OptionsManager::IsResumeModeEnable()) {
                TORCH_NPU_WARN("ACL stream synchronize failed, error code:", error,
                               ". But in checkpoint-resume mode will not throw exceptions.");
            } else {
                AT_ERROR("ACL stream synchronize failed, error code:", error);
            }
        }
        auto ret = c10_npu::acl::AclrtMemcpyBatch(dsts, dstLens, srcs, srcLens, count, attrs, attrsIndexes, count,
                                                  &failIdx);
        NPU_CHECK_ERROR(ret, "aclrtMemcpyBatch");
    }
}

void _foreach_copy_(const at::TensorList self, const at::TensorList src, bool non_blocking)
{
    DO_COMPATIBILITY(aclnnForeachCopy, at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking));
    at::native::check_foreach_api_restrictions(self, src);
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    static const bool is_support_batch = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend910_9391);

    if (!is_support_nd_out || !at::native::can_use_fast_route(self, src) || !check_tensor_dtype_support_base(src)) {
        if (is_support_batch && ((non_blocking && c10_npu::acl::IsExistMemcpyBatchAsync()) ||
                (!non_blocking && c10_npu::acl::IsExistMemcpyBatch())) && check_tensor_device_dtype_base(self, src)) {
            return memcpyBatch(self, src, non_blocking);
        }
        ASCEND_LOGW(
            "The current situation does not support the use of the memcpyBatch interface in the foreach copy interface."
            "There may be the following reasons:1.SOC version is not supported; 2.CANN version is not supported;"
            "3.The direction of the tensor devices for srcs and dsts is inconsistent,"
            "and mixed H2D and D2H scenarios are not supported."
            "For example, all tensors on the srcList must be on the host or device side");
        return at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    }

    split_and_exec_npu_cmd_copy(self, src, non_blocking);
}

#endif
} // namespace at_npu
